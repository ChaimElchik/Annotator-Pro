import torch
from PIL import Image
import numpy as np
import random
from types import SimpleNamespace

# All original imports
from util.slconfig import SLConfig
import datasets_inference.transforms as T
from models.registry import MODULE_BUILD_FUNCS

# This function is a modified version of your script's build_model_and_transforms
def load_detector_model(config_path, model_path, device_str="cuda"):
    """
    Loads the detection model and transforms once.
    """
    # We create a 'fake' args object to pass to the model builder
    args = SimpleNamespace()
    args.config = config_path
    args.pretrain_model_path = model_path
    args.device = device_str
    
    # --- This block is from your original script ---
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    data_transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])
    cfg = SLConfig.fromfile(args.config)
    # Use standard HF model ID instead of local path if possible, or make it configurable
    # If the user has it locally, we could check, but 'bert-base-uncased' is safer for general use
    cfg.merge_from_dict({"text_encoder_type": "bert-base-uncased"})
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
    
    device = torch.device(args.device)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)
    model.to(device)

    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    # --- End of original block ---
    
    print(f"Detector model '{args.modelname}' loaded to {device}.")
    return model, data_transform, device


# This function is a modified version of your script's run_inference_single_image
def run_detector_inference(model, transform, image_pil, text_prompt, device, confidence_thresh=0.23):
    """
    Runs inference on a single PIL image using the already-loaded model.
    Returns a list of YOLO-formatted boxes: [[0, xc, yc, w, h, conf], ...]
    """
    
    # 1. Transform the image
    input_image, target = transform(image_pil, {"exemplars": torch.tensor([])})
    input_image = input_image.to(device)
    input_exemplar = target["exemplars"].to(device)
    
    # 2. Run the model
    with torch.no_grad():
        output = model(
            input_image.unsqueeze(0),
            [input_exemplar],
            [torch.tensor([0]).to(device)],
            captions=[text_prompt + " ."],
        )
    
    # 3. Process outputs
    logits = output["pred_logits"][0].sigmoid()
    boxes = output["pred_boxes"][0]
    
    mask = logits.max(dim=-1).values > confidence_thresh
    logits = logits[mask, :]
    boxes = boxes[mask, :]
    conf_scores = logits.max(dim=-1).values
    pred_count = boxes.shape[0]

    if pred_count == 0:
        return []  # Return empty list if no detections

    # 4. Convert boxes to pixel coordinates
    w, h = image_pil.size
    boxes_px = boxes.clone()
    boxes_px[:, 0] *= w
    boxes_px[:, 1] *= h
    boxes_px[:, 2] *= w
    boxes_px[:, 3] *= h

    # 5. Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
    boxes_xyxy = torch.zeros_like(boxes_px)
    boxes_xyxy[:, 0] = boxes_px[:, 0] - boxes_px[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_px[:, 1] - boxes_px[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_px[:, 0] + boxes_px[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_px[:, 1] + boxes_px[:, 3] / 2

    # 6. Format as YOLO-style boxes and return
    yolo_boxes = []
    for i in range(len(boxes_xyxy)):
        box = boxes_xyxy[i]
        conf = float(conf_scores[i])
        x_min, y_min, x_max, y_max = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        
        # Convert pixel xyxy to normalized xywh
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h
        
        # Format: [class_id, x_center, y_center, width, height, confidence]
        yolo_boxes.append([0, x_center, y_center, width, height, conf])
    
    return yolo_boxes