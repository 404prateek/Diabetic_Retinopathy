"""
test_hybrid_yolo.py - Automated End-to-End Test Suite for Hybrid YOLOv8 Pipeline
================================================================================
Verifies all 4 phases of the hybrid YOLOv8 lesion detection pipeline:
  Phase 1: Dataset acquisition & YOLO formatting (label bounds & class counts)
  Phase 2: Vessel subtraction preprocessing
  Phase 3: YOLOv8 model training weights
  Phase 4: ONNX export graph structure (shapes & opset) and MATLAB integration
"""

import os
import sys
import unittest
import glob
import numpy as np

class TestHybridYOLOv8Pipeline(unittest.TestCase):
    
    def setUp(self):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.dataset_yolo = os.path.join(self.repo_root, "dataset_yolo")
        self.models_detection = os.path.join(self.repo_root, "models", "detection")
        
    def test_phase1_dataset_structure_and_yaml(self):
        """Verify dataset_yolo directory hierarchy and data.yaml configuration."""
        data_yaml = os.path.join(self.dataset_yolo, "data.yaml")
        self.assertTrue(os.path.exists(data_yaml), "dataset_yolo/data.yaml must exist.")
        
        with open(data_yaml, 'r') as f:
            content = f.read()
        self.assertIn("microaneurysm", content)
        self.assertIn("hemorrhage", content)
        self.assertIn("hard_exudate", content)
        self.assertIn("soft_exudate", content)
        self.assertIn("nc: 4", content)
        
        for split in ['train', 'val']:
            img_dir = os.path.join(self.dataset_yolo, "images", split)
            lbl_dir = os.path.join(self.dataset_yolo, "labels", split)
            self.assertTrue(os.path.exists(img_dir), f"Directory {img_dir} must exist.")
            self.assertTrue(os.path.exists(lbl_dir), f"Directory {lbl_dir} must exist.")

    def test_phase1_yolo_label_coordinate_bounds(self):
        """Verify all bounding box coordinates are strictly normalized in [0, 1]."""
        label_files = glob.glob(os.path.join(self.dataset_yolo, "labels", "**", "*.txt"), recursive=True)
        self.assertGreater(len(label_files), 0, "At least one YOLO label file must exist.")
        
        total_boxes = 0
        for l_file in label_files:
            with open(l_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                parts = line.split()
                self.assertEqual(len(parts), 5, f"Each YOLO line must have 5 tokens: {line}")
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                self.assertIn(cls_id, [0, 1, 2, 3], f"Invalid class index {cls_id}")
                self.assertTrue(0.0 <= xc <= 1.0, f"x_center {xc} out of bounds in {l_file}")
                self.assertTrue(0.0 <= yc <= 1.0, f"y_center {yc} out of bounds in {l_file}")
                self.assertTrue(0.0 < w <= 1.0, f"width {w} out of bounds in {l_file}")
                self.assertTrue(0.0 < h <= 1.0, f"height {h} out of bounds in {l_file}")
                total_boxes += 1
                
        print(f"\n[Test] Verified {total_boxes} bounding boxes with valid [0, 1] normalized coordinates.")

    def test_phase2_vessel_subtraction_scripts_exist(self):
        """Verify batch vessel subtraction scripts exist for both MATLAB and Python."""
        matlab_script = os.path.join(self.repo_root, "scripts", "vessel_subtract_batch.m")
        python_script = os.path.join(self.repo_root, "scripts", "vessel_subtract_batch.py")
        self.assertTrue(os.path.exists(matlab_script), "vessel_subtract_batch.m must exist.")
        self.assertTrue(os.path.exists(python_script), "vessel_subtract_batch.py must exist.")

    def test_phase3_training_artifacts_and_weights(self):
        """Verify best.pt weights file exists and is a valid PyTorch checkpoint."""
        import torch
        weights_path = os.path.join(self.models_detection, "best.pt")
        self.assertTrue(os.path.exists(weights_path), f"best.pt not found at {weights_path}")
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location="cpu")
        self.assertIn("model", checkpoint, "Trained checkpoint must contain model key.")
        print(f"[Test] Successfully loaded PyTorch YOLOv8 checkpoint ({os.path.getsize(weights_path)/(1024*1024):.2f} MB)")

    def test_phase4_onnx_model_specifications(self):
        """Verify best.onnx is valid, conforms to opset 12, and has expected tensor shapes."""
        import onnx
        onnx_path = os.path.join(self.models_detection, "best.onnx")
        self.assertTrue(os.path.exists(onnx_path), f"best.onnx not found at {onnx_path}")
        
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        # Verify opset version
        opset = model.opset_import[0].version
        self.assertEqual(opset, 12, f"Expected ONNX opset 12 for MATLAB compatibility, got {opset}")
        
        # Verify input & output signatures
        inputs = model.graph.input
        self.assertEqual(len(inputs), 1, "Expected single input tensor.")
        in_shape = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
        self.assertEqual(in_shape, [1, 3, 512, 512], f"Expected input shape [1, 3, 512, 512], got {in_shape}")
        
        outputs = model.graph.output
        self.assertEqual(len(outputs), 1, "Expected single output tensor.")
        out_shape = [d.dim_value for d in outputs[0].type.tensor_type.shape.dim]
        self.assertEqual(out_shape, [1, 8, 5376], f"Expected output shape [1, 8, 5376], got {out_shape}")
        print(f"[Test] ONNX validation passed! Input: {in_shape}, Output: {out_shape}, Opset: {opset}")

    def test_phase4_matlab_dashboard_integration(self):
        """Verify MATLAB scripts detectLesionsYOLO.m and runMasterPipeline.m are present and configured."""
        detector_m = os.path.join(self.repo_root, "module2_segmentation", "detectLesionsYOLO.m")
        dashboard_m = os.path.join(self.repo_root, "runMasterPipeline.m")
        
        self.assertTrue(os.path.exists(detector_m), "detectLesionsYOLO.m must exist.")
        self.assertTrue(os.path.exists(dashboard_m), "runMasterPipeline.m must exist.")
        
        with open(dashboard_m, 'r') as f:
            content = f.read()
        self.assertIn("detectLesionsYOLO", content)
        self.assertIn("subplot(1, 3, 2)", content)
        self.assertIn("YOLOv8", content)

if __name__ == "__main__":
    unittest.main()
