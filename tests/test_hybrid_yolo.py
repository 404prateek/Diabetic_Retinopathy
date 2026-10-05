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
        
    def test_dataset_structure_and_yaml(self):
        data_yaml = os.path.join(self.dataset_yolo, "data.yaml")
        self.assertTrue(os.path.exists(data_yaml))
        
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
            self.assertTrue(os.path.exists(img_dir))
            self.assertTrue(os.path.exists(lbl_dir))

    def test_yolo_label_coordinates(self):
        label_files = glob.glob(os.path.join(self.dataset_yolo, "labels", "**", "*.txt"), recursive=True)
        self.assertGreater(len(label_files), 0)
        
        for l_file in label_files:
            with open(l_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                parts = line.split()
                self.assertEqual(len(parts), 5)
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                self.assertIn(cls_id, [0, 1, 2, 3])
                self.assertTrue(0.0 <= xc <= 1.0)
                self.assertTrue(0.0 <= yc <= 1.0)
                self.assertTrue(0.0 < w <= 1.0)
                self.assertTrue(0.0 < h <= 1.0)

    def test_vessel_subtraction_scripts_exist(self):
        matlab_script = os.path.join(self.repo_root, "scripts", "vessel_subtract_batch.m")
        python_script = os.path.join(self.repo_root, "scripts", "vessel_subtract_batch.py")
        self.assertTrue(os.path.exists(matlab_script))
        self.assertTrue(os.path.exists(python_script))

    def test_trained_checkpoint(self):
        import torch
        weights_path = os.path.join(self.models_detection, "best.pt")
        self.assertTrue(os.path.exists(weights_path))
        checkpoint = torch.load(weights_path, map_location="cpu")
        self.assertIn("model", checkpoint)

    def test_onnx_model_specifications(self):
        import onnx
        onnx_path = os.path.join(self.models_detection, "best.onnx")
        self.assertTrue(os.path.exists(onnx_path))
        
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        opset = model.opset_import[0].version
        self.assertEqual(opset, 12)
        
        inputs = model.graph.input
        self.assertEqual(len(inputs), 1)
        in_shape = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
        self.assertEqual(in_shape, [1, 3, 512, 512])
        
        outputs = model.graph.output
        self.assertEqual(len(outputs), 1)
        out_shape = [d.dim_value for d in outputs[0].type.tensor_type.shape.dim]
        self.assertEqual(out_shape, [1, 8, 5376])

    def test_matlab_dashboard_integration(self):
        detector_m = os.path.join(self.repo_root, "module2_segmentation", "detectLesionsYOLO.m")
        dashboard_m = os.path.join(self.repo_root, "runMasterPipeline.m")
        
        self.assertTrue(os.path.exists(detector_m))
        self.assertTrue(os.path.exists(dashboard_m))
        
        with open(dashboard_m, 'r') as f:
            content = f.read()
        self.assertIn("detectLesionsYOLO", content)
        self.assertIn("subplot(1, 3, 2)", content)

if __name__ == "__main__":
    unittest.main()
