import os
import sys
import glob
import shutil
import argparse
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

CLASS_MAPPING = {
    'microaneurysm': 0, 'ma': 0, '1. microaneurysms': 0,
    'hemorrhage': 1, 'he': 1, 'haemorrhages': 1, '2. haemorrhages': 1,
    'hard_exudate': 2, 'ex': 2, 'hard exudates': 2, '3. hard exudates': 2,
    'soft_exudate': 3, 'se': 3, 'soft exudates': 3, '4. soft exudates': 3, 'cws': 3
}

CLASS_NAMES = ['microaneurysm', 'hemorrhage', 'hard_exudate', 'soft_exudate']

def get_bounding_boxes_from_mask(mask_path, min_area=3):
    if not os.path.exists(mask_path):
        return []
        
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    boxes = []
    
    if cv2 is not None:
        binary = (mask_np > 127).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            boxes.append((x, y, x + w, y + h))
    else:
        binary = mask_np > 127
        if not np.any(binary):
            return []
        from scipy.ndimage import label, find_objects
        labeled, num_features = label(binary)
        slices = find_objects(labeled)
        for s in slices:
            if s is None:
                continue
            ymin, ymax = s[0].start, s[0].stop
            xmin, xmax = s[1].start, s[1].stop
            area = (ymax - ymin) * (xmax - xmin)
            if area >= min_area:
                boxes.append((xmin, ymin, xmax, ymax))
                
    return boxes

def convert_box_to_yolo(box, img_w, img_h):
    xmin, ymin, xmax, ymax = box
    
    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(1, min(xmax, img_w))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(1, min(ymax, img_h))
    
    xc = ((xmin + xmax) / 2.0) / img_w
    yc = ((ymin + ymax) / 2.0) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    bw = max(0.001, min(1.0, bw))
    bh = max(0.001, min(1.0, bh))
    
    return xc, yc, bw, bh

def find_mask_for_lesion(idrid_dir, image_id, set_type, lesion_key):
    lesion_patterns = {
        'ma': ['*Microaneurysms*', '*MA*', '*ma*'],
        'he': ['*Haemorrhages*', '*HE*', '*he*', '*Hemorrhages*'],
        'ex': ['*Hard*Exudates*', '*EX*', '*ex*'],
        'se': ['*Soft*Exudates*', '*SE*', '*se*']
    }
    
    patterns = lesion_patterns.get(lesion_key, [])
    search_dirs = [
        os.path.join(idrid_dir, "2. Groundtruths", set_type),
        os.path.join(idrid_dir, "Groundtruths", set_type),
        os.path.join(idrid_dir, "masks", set_type),
        os.path.join(idrid_dir, "2. All Segmentation Groundtruths"),
        os.path.join(idrid_dir, "Groundtruths"),
        idrid_dir
    ]
    
    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            continue
        for root, dirs, files in os.walk(s_dir):
            for f in files:
                f_lower = f.lower()
                img_id_lower = image_id.lower()
                if img_id_lower in f_lower:
                    for pat in patterns:
                        pat_clean = pat.replace('*', '').lower()
                        if pat_clean in f_lower or pat_clean in root.lower():
                            return os.path.join(root, f)
                            
    return None

def process_idrid_dataset(idrid_dir, output_dir, val_split=0.2):
    idrid_dir = os.path.abspath(idrid_dir)
    output_dir = os.path.abspath(output_dir)
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        all_images.extend(glob.glob(os.path.join(idrid_dir, '**', ext), recursive=True))
        
    raw_images = []
    for img_p in all_images:
        f_upper = os.path.basename(img_p).upper()
        if any(gt_tag in f_upper for gt_tag in ['_MA', '_HE', '_EX', '_SE', 'MASK', 'GROUNDTRUTH', '_OD', '_CC']):
            continue
        raw_images.append(img_p)
        
    raw_images = sorted(list(set(raw_images)))
    if len(raw_images) == 0:
        print("No raw fundus images found.")
        return False
        
    np.random.seed(42)
    indices = np.random.permutation(len(raw_images))
    val_count = max(1, int(len(raw_images) * val_split))
    val_indices = set(indices[:val_count])
    
    total_boxes = 0
    class_box_counts = {c: 0 for c in range(4)}
    
    for idx, img_path in enumerate(raw_images):
        split = 'val' if idx in val_indices else 'train'
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        
        with Image.open(img_path) as im:
            img_w, img_h = im.size
            
        dest_img_path = os.path.join(output_dir, 'images', split, f"{img_basename}.jpg")
        shutil.copy2(img_path, dest_img_path)
        
        yolo_annotations = []
        set_type = "b. Testing Set" if "test" in img_path.lower() else "a. Training Set"
        
        for lesion_name, class_id in [('ma', 0), ('he', 1), ('ex', 2), ('se', 3)]:
            mask_path = find_mask_for_lesion(idrid_dir, img_basename, set_type, lesion_name)
            if mask_path and os.path.exists(mask_path):
                boxes = get_bounding_boxes_from_mask(mask_path)
                for box in boxes:
                    xc, yc, bw, bh = convert_box_to_yolo(box, img_w, img_h)
                    yolo_annotations.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    class_box_counts[class_id] += 1
                    total_boxes += 1
                    
        label_file = os.path.join(output_dir, 'labels', split, f"{img_basename}.txt")
        with open(label_file, 'w') as lf:
            if yolo_annotations:
                lf.write("\n".join(yolo_annotations) + "\n")
            
    print(f"Extracted {total_boxes} boxes across {len(raw_images)} images.")
    create_data_yaml(output_dir)
    return True

def create_data_yaml(dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    clean_dir = os.path.abspath(dataset_dir).replace('\\', '/')
    
    yaml_content = f"""path: {clean_dir}
train: images/train
val: images/val

nc: 4
names:
  0: microaneurysm
  1: hemorrhage
  2: hard_exudate
  3: soft_exudate
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

def main():
    parser = argparse.ArgumentParser(description="Convert IDRiD lesion annotations to YOLO format.")
    parser.add_argument("--idrid-dir", default="data/IDRiD")
    parser.add_argument("--output-dir", default="dataset_yolo")
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()
    
    success = process_idrid_dataset(args.idrid_dir, args.output_dir, args.val_split)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
