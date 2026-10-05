"""
vessel_subtract_batch.py - Phase 2: Batch Retinal Vessel Subtraction (Python)
=============================================================================
Preprocesses retinal images by segmenting blood vessels and applying
OpenCV Telea inpainting or background blending to create vessel-free images
for YOLOv8 training.
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

def detect_vessels_morphological(img_bgr):
    """Detects vessels from green channel using morphological top-hat filtering."""
    green = img_bgr[:, :, 1]
    
    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    
    # Inverted morphological top-hat (extract dark vessel structures)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(255 - enhanced, cv2.MORPH_TOPHAT, kernel)
    
    # Adaptive threshold
    thresh_val = np.percentile(tophat, 88)
    vessel_mask = (tophat > thresh_val).astype(np.uint8) * 255
    
    # Remove small speckles
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return vessel_mask

def inpaint_vessels(img_bgr, mask):
    """Inpaints vessel pixels with surrounding retinal texture using Telea algorithm."""
    # Dilate mask slightly to prevent dark boundary artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    
    inpainted = cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    return inpainted

def process_directory(img_dir, backup=False):
    """Processes all images in the target directory."""
    if cv2 is None:
        print("[!] OpenCV is required for inpainting. Please ensure opencv-python is installed.")
        return
        
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
        
    img_files = sorted(img_files)
    print(f"[*] Processing {len(img_files)} images in: {img_dir}")
    
    for idx, path in enumerate(img_files):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
            
        vessel_mask = detect_vessels_morphological(img_bgr)
        cleaned = inpaint_vessels(img_bgr, vessel_mask)
        
        if backup:
            bak_path = path + ".raw.bak"
            if not os.path.exists(bak_path):
                os.rename(path, bak_path)
                
        cv2.imwrite(path, cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        if (idx + 1) % 10 == 0 or idx == len(img_files) - 1:
            print(f"    [{idx+1}/{len(img_files)}] Subtracted vessels from {os.path.basename(path)}")

def main():
    parser = argparse.ArgumentParser(description="Batch vessel subtraction for retinal lesion training.")
    parser.add_argument("--data-dir", default="dataset_yolo", help="YOLO dataset root containing images/train and images/val")
    parser.add_argument("--backup", action="store_true", help="Keep backup of original images")
    args = parser.parse_args()
    
    for split in ['train', 'val']:
        folder = os.path.join(args.data_dir, 'images', split)
        if os.path.exists(folder):
            process_directory(folder, backup=args.backup)

if __name__ == "__main__":
    main()
