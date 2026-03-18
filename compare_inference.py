import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
from rfdetr import RFDETRMedium
from detector_wrapper import DetectorWrapper

# 1. Setup paths
img_path = 'subset_10_percent_COCO/test/AEWD-main_coco_cleaned_Amur Tiger_103.jpg'
model_path = 'data/models/checkpoint_best_ema.pth'

if not os.path.exists(img_path):
    print(f"Error: Image {img_path} not found")
    sys.exit(1)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 2. Run like Single_Image_Inference_Options_Og.py (User style)
print("\n--- Running User Style Inference (OG Script) ---")
try:
    # Based on OG script load_rf_detr
    model_og = RFDETRMedium(
        pretrain_weights=model_path,
        resolution=640,
        num_classes=2, # OG script used 2
        device=device
    )
    model_og.optimize_for_inference()
    
    image_bgr = cv2.imread(img_path)
    image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    
    det_og = model_og.predict(image_pil, threshold=0.1)
    print(f"OG Script Detections: {len(det_og)}")
    if len(det_og) > 0:
        for i in range(len(det_og.class_id)):
            print(f"  [{i}] CID: {det_og.class_id[i]}, Conf: {det_og.confidence[i]:.4f}")
except Exception as e:
    print(f"OG Style failed: {e}")

# 3. Run like DetectorWrapper (Tool style)
print("\n--- Running Tool Style Inference (DetectorWrapper) ---")
try:
    dw = DetectorWrapper()
    # Mocking the run_inference slightly to print raw data before mapping
    # We can just look at what it returns now
    results_tool = dw.run_inference(img_path, model_type='rfdetr', model_path=model_path, confidence=0.1)
    print(f"Tool Detections: {len(results_tool)}")
    for i, res in enumerate(results_tool):
        print(f"  [{i}] Label: {res['label']}, Conf: {res['confidence']:.4f}")
except Exception as e:
    print(f"Tool Style failed: {e}")
