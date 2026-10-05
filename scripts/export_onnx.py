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
    
    if not os.path.exists(weights_path):
        alt_path = os.path.abspath("runs/detect/train/weights/best.pt")
        weights_path = alt_path if os.path.exists(alt_path) else "yolov8s.pt"
            
    yolo_model = YOLO(weights_path)
    
    local_target_dir = os.path.abspath("models/detection")
    os.makedirs(local_target_dir, exist_ok=True)
    local_target = os.path.join(local_target_dir, "best.onnx")
    
    torch_model = yolo_model.model
    torch_model.eval()
    
    # Configure detect layer to export single concatenated tensor
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
    
    print(f"Exported ONNX model to {local_target}")
    
    if os.path.exists(sih_sync_dir):
        sih_dest = os.path.join(sih_sync_dir, "models", "detection")
        os.makedirs(sih_dest, exist_ok=True)
        shutil.copy2(local_target, os.path.join(sih_dest, "best.onnx"))
        print(f"Copied to {sih_sync_dir}")
        
    return local_target

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX for MATLAB.")
    parser.add_argument("--weights", default="models/detection/best.pt")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--sih-path", default=DEFAULT_SIH_DIR)
    args = parser.parse_args()
    
    export_model_to_onnx(
        weights_path=args.weights,
        img_size=args.imgsz,
        opset=args.opset,
        sih_sync_dir=args.sih_path
    )

if __name__ == "__main__":
    main()
