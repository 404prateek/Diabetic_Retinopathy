"""
export_onnx.py - Phase 4: Export YOLOv8 to ONNX for MATLAB Integration
======================================================================
Purpose:
  Converts trained YOLOv8 PyTorch model (.pt) into cross-platform ONNX (.onnx)
  configured specifically for MATLAB Deep Learning Toolbox importNetworkFromONNX.
  Produces a clean single-output tensor [1, 4 + nc, 5376] (e.g. [1, 8, 5376]).
  Synchronizes model to both local models/detection/ and C:/SIH_Work/Diabetic_Retinopathy.

Usage:
  python scripts/export_onnx.py --weights models/detection/best.pt --imgsz 512
"""

import os
import sys
import shutil
import argparse
import torch

DEFAULT_SIH_DIR = "C:/SIH_Work/Diabetic_Retinopathy"

def export_model_to_onnx(weights_path="models/detection/best.pt",
                         img_size=512,
                         opset=12,
                         sih_sync_dir=DEFAULT_SIH_DIR):
    
    from ultralytics import YOLO
    
    weights_path = os.path.abspath(weights_path)
    
    # Fallback to runs/detect/train/weights/best.pt or base yolov8s.pt
    if not os.path.exists(weights_path):
        alt_path = os.path.abspath("runs/detect/train/weights/best.pt")
        if os.path.exists(alt_path):
            weights_path = alt_path
        else:
            print(f"[!] Target weights not found at {weights_path}.")
            print("[*] Falling back to base yolov8s.pt for verification...")
            weights_path = "yolov8s.pt"
            
    print(f"[*] Loading model for ONNX export: {weights_path}...")
    yolo_model = YOLO(weights_path)
    
    local_target_dir = os.path.abspath("models/detection")
    os.makedirs(local_target_dir, exist_ok=True)
    local_target = os.path.join(local_target_dir, "best.onnx")
    
    print(f"[*] Configuring YOLOv8 Detect head for static fused tensor (imgsz={img_size}, opset={opset})...")
    
    torch_model = yolo_model.model
    torch_model.eval()
    
    # Enable export mode on the Detect head so it fuses into [1, 4+nc, 5376]
    if hasattr(torch_model.model[-1], 'export'):
        torch_model.model[-1].export = True
        torch_model.model[-1].format = 'onnx'
        
    dummy_input = torch.zeros(1, 3, img_size, img_size)
    
    torch.onnx.export(
        torch_model,
        dummy_input,
        local_target,
        opset_version=opset,
        input_names=['images'],
        output_names=['output0'],
        dynamic_axes=None
    )
    
    print(f"[+] Direct ONNX export successful: {local_target}")
    
    # Copy to SIH work folder if present
    if os.path.exists(sih_sync_dir):
        sih_dest = os.path.join(sih_sync_dir, "models", "detection")
        os.makedirs(sih_dest, exist_ok=True)
        shutil.copy2(local_target, os.path.join(sih_dest, "best.onnx"))
        print(f"[+] Synchronized best.onnx to: {os.path.join(sih_dest, 'best.onnx')}")
    else:
        print(f"[*] Note: SIH project folder '{sih_sync_dir}' not detected on this machine. Ready for manual deployment.")
        
    return local_target

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX for MATLAB.")
    parser.add_argument("--weights", default="models/detection/best.pt", help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image dimension (default: 512)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version (default: 12)")
    parser.add_argument("--sih-path", default=DEFAULT_SIH_DIR, help="Path to SIH project folder")
    args = parser.parse_args()
    
    export_model_to_onnx(
        weights_path=args.weights,
        img_size=args.imgsz,
        opset=args.opset,
        sih_sync_dir=args.sih_path
    )

if __name__ == "__main__":
    main()
