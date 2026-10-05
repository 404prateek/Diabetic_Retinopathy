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
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
        
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
            
    model = YOLO(model_type)
    output_dir = os.path.abspath("runs/detect")
    
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=2,
        project=output_dir,
        name="train",
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True
    )
    
    best_weights = os.path.join(output_dir, "train", "weights", "best.pt")
    if os.path.exists(best_weights):
        models_dir = os.path.abspath("models/detection")
        os.makedirs(models_dir, exist_ok=True)
        target_path = os.path.join(models_dir, "best.pt")
        shutil.copy2(best_weights, target_path)
        print(f"Saved weights to {target_path}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on retinal lesions.")
    parser.add_argument("--data", default="dataset_yolo/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--device", default=None)
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
