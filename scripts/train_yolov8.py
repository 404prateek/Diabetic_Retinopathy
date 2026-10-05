"""
train_yolov8.py - Phase 3: YOLOv8s Training for Retinal Lesion Detection
========================================================================
Purpose:
  Trains YOLOv8s (Small) on vessel-subtracted retinal images.
  Configured with batch_size=8 and imgsz=512 for optimal feature extraction
  while operating safely within the 4 GB VRAM ceiling of an NVIDIA RTX 3050.

Usage:
  python scripts/train_yolov8.py --epochs 50 --batch 8 --imgsz 512
"""

import os
import sys
import shutil
import argparse
import torch

def run_training(data_yaml="dataset_yolo/data.yaml",
                 epochs=50,
                 batch_size=8,
                 img_size=512,
                 model_type="yolov8s.pt",
                 device=None):
    
    from ultralytics import YOLO
    
    data_yaml_path = os.path.abspath(data_yaml)
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}. Run Phase 1 conversion first.")
        
    # Auto-detect compute hardware
    if device is None:
        if torch.cuda.is_available():
            device = 0
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[+] CUDA GPU Detected: {gpu_name} ({total_vram:.2f} GB VRAM)")
            print(f"[+] Applying RTX 3050 optimized settings: batch={batch_size}, imgsz={img_size}")
        else:
            device = "cpu"
            print("[!] CUDA GPU not detected. Running on CPU.")
            
    print(f"[*] Initializing YOLOv8 model: {model_type}...")
    model = YOLO(model_type)
    
    output_dir = os.path.abspath("runs/detect")
    
    print(f"[*] Commencing training for {epochs} epochs...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=2,            # Safe for Windows multiprocessing
        project=output_dir,
        name="train",
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True
    )
    
    # Path to saved weights
    best_weights = os.path.join(output_dir, "train", "weights", "best.pt")
    if os.path.exists(best_weights):
        models_dir = os.path.abspath("models/detection")
        os.makedirs(models_dir, exist_ok=True)
        target_path = os.path.join(models_dir, "best.pt")
        shutil.copy2(best_weights, target_path)
        print(f"[+] Successfully saved best weights to: {target_path}")
    else:
        print(f"[!] Warning: Expected best weights at {best_weights} not found.")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8s for Diabetic Retinopathy Lesion Detection.")
    parser.add_argument("--data", default="dataset_yolo/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (8 for RTX 3050 4GB)")
    parser.add_argument("--imgsz", type=int, default=512, help="Image resolution (512x512)")
    parser.add_argument("--model", default="yolov8s.pt", help="Pretrained model weights")
    parser.add_argument("--device", default=None, help="Device ('0', 'cpu', etc.)")
    args = parser.parse_args()
    
    run_training(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        model_type=args.model,
        device=args.device
    )

if __name__ == "__main__":
    main()
