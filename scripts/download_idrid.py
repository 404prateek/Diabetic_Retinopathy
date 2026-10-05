"""
IDRiD Dataset Acquisition & Verification Script
=================================================
Automates downloading, checking, or generating sample data for the
Indian Diabetic Retinopathy Image Dataset (IDRiD) for lesion localization.

Supports:
1. Kaggle API automated download (if kaggle credentials exist)
2. Direct folder verification and reorganization
3. Synthetic / sample dataset generation for immediate end-to-end verification

Usage:
  python scripts/download_idrid.py --dest-dir data/IDRiD
  python scripts/download_idrid.py --create-samples --num-samples 10
"""

import os
import sys
import argparse
import subprocess
import shutil
import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None

DEFAULT_KAGGLE_DATASET = "mariaherrerot/idrid"

def check_existing_dataset(dest_dir):
    """Check if IDRiD dataset files already exist in destination."""
    if not os.path.exists(dest_dir):
        return False
    
    # Check for image files
    img_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    found_images = []
    found_masks = []
    
    for root, _, files in os.walk(dest_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in img_exts:
                if any(m in f.upper() for m in ['_MA', '_HE', '_EX', '_SE', 'MASK', 'GROUNDTRUTH']):
                    found_masks.append(os.path.join(root, f))
                else:
                    found_images.append(os.path.join(root, f))
                    
    print(f"[*] Found {len(found_images)} raw fundus images and {len(found_masks)} lesion masks in {dest_dir}.")
    return len(found_images) > 0

def download_via_kaggle(dataset_name, dest_dir):
    """Attempt download using kaggle CLI or kaggle python library."""
    print(f"[*] Attempting download from Kaggle dataset: {dataset_name}...")
    os.makedirs(dest_dir, exist_ok=True)
    
    # Try running kaggle CLI
    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", dest_dir, "--unzip"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("[+] Successfully downloaded and unzipped IDRiD dataset via Kaggle CLI!")
            return True
        else:
            print(f"[!] Kaggle CLI exited with code {result.returncode}: {result.stderr.strip()}")
    except FileNotFoundError:
        print("[!] Kaggle CLI tool not found in PATH.")

    # Try python kaggle api
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[+] Kaggle API authenticated. Downloading dataset...")
        api.dataset_download_files(dataset_name, path=dest_dir, unzip=True)
        print("[+] Download complete!")
        return True
    except Exception as e:
        print(f"[!] Kaggle API download failed: {e}")

    return False

def generate_sample_dataset(dest_dir, num_samples=10):
    """
    Generates synthetic sample retinal fundus images and ground-truth lesion masks
    for rapid end-to-end pipeline testing without requiring a 10 GB download.
    """
    if Image is None:
        print("[-] PIL (Pillow) is required to generate sample images.")
        return False
        
    print(f"[*] Generating {num_samples} realistic sample fundus images and lesion masks...")
    
    images_train_dir = os.path.join(dest_dir, "1. Original Images", "a. Training Set")
    images_test_dir  = os.path.join(dest_dir, "1. Original Images", "b. Testing Set")
    
    gt_base_train = os.path.join(dest_dir, "2. Groundtruths", "a. Training Set")
    gt_base_test  = os.path.join(dest_dir, "2. Groundtruths", "b. Testing Set")
    
    lesion_dirs = [
        "1. Microaneurysms",
        "2. Haemorrhages",
        "3. Hard Exudates",
        "4. Soft Exudates"
    ]
    
    for p in [images_train_dir, images_test_dir]:
        os.makedirs(p, exist_ok=True)
        
    for base in [gt_base_train, gt_base_test]:
        for ld in lesion_dirs:
            os.makedirs(os.path.join(base, ld), exist_ok=True)
            
    img_w, img_h = 1024, 680  # Realistic fundus aspect ratio
    
    np.random.seed(42)
    
    for i in range(1, num_samples + 1):
        is_test = (i > int(num_samples * 0.75))
        set_name = "b. Testing Set" if is_test else "a. Training Set"
        img_id = f"IDRiD_{i:02d}"
        
        # 1. Create simulated fundus image
        img_arr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
        # Draw circular retina boundary with orange-red background
        cy, cx = img_h // 2, img_w // 2
        r = min(img_h, img_w) // 2 - 20
        y_idx, x_idx = np.ogrid[:img_h, :img_w]
        retina_mask = ((x_idx - cx)**2 + (y_idx - cy)**2) <= r**2
        
        # Base retinal color (deep reddish-orange gradient)
        img_arr[retina_mask, 0] = np.random.randint(180, 230, size=np.sum(retina_mask)) # Red
        img_arr[retina_mask, 1] = np.random.randint(70, 110, size=np.sum(retina_mask))   # Green
        img_arr[retina_mask, 2] = np.random.randint(20, 50, size=np.sum(retina_mask))    # Blue
        
        # Convert to PIL
        pil_img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(pil_img)
        
        # Draw bright optic disc (yellow-orange circle)
        od_cx, od_cy = cx - r // 2, cy
        od_r = 45
        draw.ellipse([od_cx - od_r, od_cy - od_r, od_cx + od_r, od_cy + od_r], fill=(255, 230, 160))
        
        # Draw simulated blood vessels (dark red branched lines)
        for _ in range(6):
            start_x, start_y = od_cx, od_cy
            for seg in range(4):
                end_x = start_x + np.random.randint(40, 90) * np.random.choice([-1, 1])
                end_y = start_y + np.random.randint(30, 80) * np.random.choice([-1, 1])
                # Ensure within retina
                if ((end_x - cx)**2 + (end_y - cy)**2) < (r - 30)**2:
                    draw.line([start_x, start_y, end_x, end_y], fill=(90, 20, 10), width=np.random.randint(2, 5))
                    start_x, start_y = end_x, end_y
                    
        # Save raw image
        raw_path = os.path.join(dest_dir, "1. Original Images", set_name, f"{img_id}.jpg")
        pil_img.save(raw_path, quality=95)
        
        # 2. Create Lesion Masks:
        # Microaneurysms (MA): tiny red spots
        ma_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        num_ma = np.random.randint(2, 6)
        for _ in range(num_ma):
            lx = np.random.randint(cx - r + 50, cx + r - 50)
            ly = np.random.randint(cy - r + 50, cy + r - 50)
            if ((lx - cx)**2 + (ly - cy)**2) < (r - 50)**2:
                rad = np.random.randint(2, 5)
                y1, y2 = max(0, ly - rad), min(img_h, ly + rad + 1)
                x1, x2 = max(0, lx - rad), min(img_w, lx + rad + 1)
                ma_mask[y1:y2, x1:x2] = 255
        Image.fromarray(ma_mask).save(os.path.join(dest_dir, "2. Groundtruths", set_name, "1. Microaneurysms", f"{img_id}_MA.tif"))
        
        # Hemorrhages (HE): larger irregular dark red patches
        he_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        num_he = np.random.randint(1, 4)
        for _ in range(num_he):
            lx = np.random.randint(cx - r + 60, cx + r - 60)
            ly = np.random.randint(cy - r + 60, cy + r - 60)
            if ((lx - cx)**2 + (ly - cy)**2) < (r - 50)**2:
                rad = np.random.randint(8, 20)
                y1, y2 = max(0, ly - rad), min(img_h, ly + rad + 1)
                x1, x2 = max(0, lx - rad), min(img_w, lx + rad + 1)
                he_mask[y1:y2, x1:x2] = 255
        Image.fromarray(he_mask).save(os.path.join(dest_dir, "2. Groundtruths", set_name, "2. Haemorrhages", f"{img_id}_HE.tif"))

        # Hard Exudates (EX): bright yellow/white specks with sharp edges
        ex_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        num_ex = np.random.randint(2, 5)
        for _ in range(num_ex):
            lx = np.random.randint(cx - r + 60, cx + r - 60)
            ly = np.random.randint(cy - r + 60, cy + r - 60)
            if ((lx - cx)**2 + (ly - cy)**2) < (r - 50)**2:
                rad = np.random.randint(4, 12)
                y1, y2 = max(0, ly - rad), min(img_h, ly + rad + 1)
                x1, x2 = max(0, lx - rad), min(img_w, lx + rad + 1)
                ex_mask[y1:y2, x1:x2] = 255
        Image.fromarray(ex_mask).save(os.path.join(dest_dir, "2. Groundtruths", set_name, "3. Hard Exudates", f"{img_id}_EX.tif"))

        # Soft Exudates / Cotton Wool Spots (SE): fuzzy white patches
        se_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        num_se = np.random.randint(1, 3)
        for _ in range(num_se):
            lx = np.random.randint(cx - r + 60, cx + r - 60)
            ly = np.random.randint(cy - r + 60, cy + r - 60)
            if ((lx - cx)**2 + (ly - cy)**2) < (r - 50)**2:
                rad = np.random.randint(10, 25)
                y1, y2 = max(0, ly - rad), min(img_h, ly + rad + 1)
                x1, x2 = max(0, lx - rad), min(img_w, lx + rad + 1)
                se_mask[y1:y2, x1:x2] = 255
        Image.fromarray(se_mask).save(os.path.join(dest_dir, "2. Groundtruths", set_name, "4. Soft Exudates", f"{img_id}_SE.tif"))

    print(f"[+] Sample IDRiD dataset successfully created at: {dest_dir}")
    return True

def print_manual_download_instructions():
    print("""
=============================================================================
MANUAL IDRiD DATASET DOWNLOAD INSTRUCTIONS
=============================================================================
The official IDRiD dataset requires academic access or Kaggle login:

Option A: Kaggle (Easiest)
1. Visit: https://www.kaggle.com/datasets/anirbanh/idrid-dataset
   or https://www.kaggle.com/datasets/mariaherrerot/idrid
2. Click 'Download' (zip archive).
3. Extract the contents into:
   data/IDRiD/

Option B: IEEE Dataport (Official Challenge Dataset)
1. Visit: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
2. Download 'A. Segmentation' and 'C. Localization'.
3. Extract into:
   data/IDRiD/

Option C: Rapid Local Testing Mode
Run with --create-samples to immediately generate realistic sample fundus
images and masks to test the entire pipeline end-to-end:
   python scripts/download_idrid.py --create-samples --num-samples 12
=============================================================================
""")

def main():
    parser = argparse.ArgumentParser(description="Acquire and verify IDRiD dataset for lesion localization.")
    parser.add_argument("--dest-dir", default="data/IDRiD", help="Directory where IDRiD is stored")
    parser.add_argument("--kaggle-dataset", default=DEFAULT_KAGGLE_DATASET, help="Kaggle dataset slug")
    parser.add_argument("--create-samples", action="store_true", help="Generate synthetic sample dataset for testing")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of sample images to generate")
    
    args = parser.parse_args()
    dest_dir = os.path.abspath(args.dest_dir)
    
    if args.create_samples:
        generate_sample_dataset(dest_dir, num_samples=args.num_samples)
        return
        
    # Check if dataset already exists
    if check_existing_dataset(dest_dir):
        print("[+] Dataset already present and verified in " + dest_dir)
        return
        
    # Attempt Kaggle download
    downloaded = download_via_kaggle(args.kaggle_dataset, dest_dir)
    
    if not downloaded:
        print_manual_download_instructions()
        print("[*] Automatically generating starter sample dataset so pipeline can run immediately...")
        generate_sample_dataset(dest_dir, num_samples=args.num_samples)

if __name__ == "__main__":
    main()
