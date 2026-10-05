import os
import glob
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

def detect_vessels(img_bgr):
    green = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(255 - enhanced, cv2.MORPH_TOPHAT, kernel)
    
    thresh_val = np.percentile(tophat, 88)
    vessel_mask = (tophat > thresh_val).astype(np.uint8) * 255
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return vessel_mask

def inpaint_vessels(img_bgr, mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

def process_directory(img_dir, backup=False):
    if cv2 is None:
        print("OpenCV not installed.")
        return
        
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
        
    img_files = sorted(img_files)
    print(f"Processing {len(img_files)} images in {img_dir}...")
    
    for idx, path in enumerate(img_files):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
            
        mask = detect_vessels(img_bgr)
        cleaned = inpaint_vessels(img_bgr, mask)
        
        if backup:
            bak_path = path + ".raw.bak"
            if not os.path.exists(bak_path):
                os.rename(path, bak_path)
                
        cv2.imwrite(path, cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def main():
    parser = argparse.ArgumentParser(description="Batch vessel inpainting on retinal images.")
    parser.add_argument("--data-dir", default="dataset_yolo")
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()
    
    for split in ['train', 'val']:
        folder = os.path.join(args.data_dir, 'images', split)
        if os.path.exists(folder):
            process_directory(folder, backup=args.backup)

if __name__ == "__main__":
    main()
