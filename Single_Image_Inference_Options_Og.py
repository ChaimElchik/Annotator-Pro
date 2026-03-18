import supervision as sv
import cv2
import numpy as np
from PIL import Image
import os
import torch
import cv2
import numpy as np
import supervision as sv
from PIL import Image
from ultralytics import YOLO
from rfdetr import RFDETRMedium
import sys
from functools import partial

def rf_detr_callback(image_slice: np.ndarray, model) -> sv.Detections:
    slice_pil = Image.fromarray(cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB))
    return model.predict(slice_pil, threshold=0.5)


def Run_Inference_Single_Image(img_name, rf_model):
    image_bgr = cv2.imread(img_name)
    image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # --- A. Run Standard Model First ---
    det_std = rf_model.predict(image_pil, threshold=0.5)
    
    use_combined = False
    
    # Logic for switching to Tiled Inference
    if len(det_std) == 0:
        use_combined = True
        # print("slicing: ", img_name)
        # use_combined = False
    else:
        h, w = image_bgr.shape[:2]
        image_area = h * w
        box_areas = (det_std.xyxy[:, 2] - det_std.xyxy[:, 0]) * (det_std.xyxy[:, 3] - det_std.xyxy[:, 1])
        avg_rel_area = np.mean(box_areas / image_area)
        
        if avg_rel_area <= 0.000326:
        # if avg_rel_area <= 0.00000000300:
            # print("slicing: ", img_name)
            use_combined = True

    # --- B. Execute Final Pipeline ---
    if use_combined:
        # We create the slicer HERE, injecting the specific model instance
        # using functools.partial to wrap the callback
        callback_with_model = partial(rf_detr_callback, model=rf_model)
        
        rf_slicer = sv.InferenceSlicer(
            callback=callback_with_model,
            slice_wh=(640, 640),
            iou_threshold=0.5
        )
        det_final = rf_slicer(image_bgr)
    else:
        det_final = det_std

    return det_final
# Viz functions

def create_view(img, dets, title, class_names=None, color=(0, 255, 0)):
    """
    Visualizes detections with a title bar, class names, and confidence labels.
    """
    # 1. Setup Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
    text_scale=0.6, 
    text_color=sv.Color.BLACK,
    text_position=sv.Position.CENTER  # Draws the label right in the middle of the box
    )

    # 2. Apply Annotations
    res = img.copy()
    res = box_annotator.annotate(scene=res, detections=dets)
    
    # Generate labels (Class Name + Confidence score)
    if class_names is not None and dets.class_id is not None:
        labels = [
            f"{class_names[class_id]} {conf:.2f}" 
            for class_id, conf in zip(dets.class_id, dets.confidence)
        ]
    else:
        # Fallback to just confidence if no class names are provided
        labels = [f"{c:.2f}" for c in dets.confidence]
        
    res = label_annotator.annotate(scene=res, detections=dets, labels=labels)
    
    # 3. Add Header Bar
    h, w = res.shape[:2]
    cv2.rectangle(res, (0, 0), (w, 60), (0, 0, 0), -1)
    
    # 4. Add Title Text
    text = f"{title}: {len(dets)} Boxes"
    cv2.putText(res, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return res


def load_rf_detr(model_path, device):
    print(f"Loading RF-DETR model from {model_path} on {device}...")
    try:
        # Checkpoint has 2 classes based on error message
        model = RFDETRMedium(
            pretrain_weights=model_path,
            resolution=640,
            num_classes=2,
            device=device
        )
        model.optimize_for_inference()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    img_name = "subset_10_percent_COCO/test/Songdo Vision_coco_cleaned_04674.jpg"
    image_bgr = cv2.imread(img_name)
    image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    
    DEFAULT_MODEL = "training_outputs/checkpoint_best_ema.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Define your class names here. Since num_classes=2, you need 2 string elements.
    # Replace these with your actual class names (e.g., ["Cattle", "Other"])
    CLASS_NAMES = ["Class_0", "Class_1"] 
    
    rf_model = load_rf_detr(DEFAULT_MODEL, DEVICE)
    det_final = Run_Inference_Single_Image(img_name, rf_model)

    # --- D. Visualize Result ---
    # Pass the CLASS_NAMES into your updated create_view function
    res = create_view(image_bgr, det_final, "Final Result", class_names=CLASS_NAMES, color=(0, 255, 0))
    
    # cv2.imwrite("result.jpg", res)
    cv2.imshow("Result", res)
    cv2.waitKey(0)