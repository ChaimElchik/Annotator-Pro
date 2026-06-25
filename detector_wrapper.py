import os
import torch
import detector_logic
from PIL import Image
import uuid
import cv2
import numpy as np

# Try imports
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. YOLO support disabled.")
    YOLO = None

try:
    from rfdetr.variants import RFDETRMedium
except ImportError:
    RFDETRMedium = None

try:
    import supervision as sv
except ImportError:
    print("Warning: supervision not installed. Tiled RF-DETR support disabled.")
    sv = None

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError:
    print("Warning: sahi not installed. Tiled YOLO support disabled.")
    AutoDetectionModel = None

from functools import partial

class DetectorWrapper:
    _instance = None
    
    def __init__(self):
        # CountGD
        self.countgd_model = None
        self.countgd_transform = None
        self.countgd_device = None
        
        # Grounding DINO
        self.grounding_dino_processor = None
        self.grounding_dino_model = None

        # Cache for other models: path -> model_instance
        self.model_cache = {}
        
        # Robust path finding for PyInstaller
        import sys
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        elif getattr(sys, 'frozen', False):
            base_path = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))

        self.config_path = os.path.join(base_path, "config", "cfg_fsc147_vit_b.py")
        self.checkpoint_path = os.path.join(base_path, "checkpoint_fsc147_best.pth")
        
        # Determine optimal device
        if torch.backends.mps.is_available():
            self.device_str = "mps"
        elif torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"
            
        print(f"DetectorWrapper initialized. Default Device: {self.device_str}")
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def load_countgd(self):
        if self.countgd_model is None:
            if not os.path.exists(self.config_path) or not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError("CountGD config or checkpoint not found.")
                
            print("Loading CountGD model...")
            self.countgd_model, self.countgd_transform, self.countgd_device = detector_logic.load_detector_model(
                self.config_path, 
                self.checkpoint_path, 
                device_str=self.device_str
            )
            print(f"CountGD Loaded on {self.countgd_device}")

    def load_yolo(self, weights_path):
        if weights_path in self.model_cache:
            return self.model_cache[weights_path]
            
        if YOLO is None:
             # Critical error if user tries to load YOLO and it's missing
            raise ImportError("The 'ultralytics' library is not installed. Please install it with `pip install ultralytics` to use YOLO models.")
            
        print(f"Loading YOLO from {weights_path}...")
        try:
            model = YOLO(weights_path)
            self.model_cache[weights_path] = model
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model. If this is a valid YOLO model, PyTorch may be failing to unpickle it (e.g. weights_only=True restriction or corrupted file). Details: {e}")

    def load_sam2(self, weights_path):
        if weights_path in self.model_cache:
            return self.model_cache[weights_path]
            
        if YOLO is None: # Since SAM belongs to ultralytics
             raise ImportError("The 'ultralytics' library is not installed.")
             
        print(f"Loading SAM2 from {weights_path}...")
        try:
            from ultralytics import SAM
            model = SAM(weights_path)
            self.model_cache[weights_path] = model
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2 model: {e}")

    def load_grounding_dino(self):
        if self.grounding_dino_model is None:
            print("Loading Grounding DINO (Transformers)...")
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.grounding_dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
            self.grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
            # Usually safe to run HF models on CUDA/MPS
            self.grounding_dino_model.to(self.device_str)
            print(f"Grounding DINO Loaded on {self.device_str}")

    def load_sam3(self, weights_path, confidence=0.25):
        # We instantiate predictably because SAM3SemanticPredictor binds to the selected confidence
        if weights_path in self.model_cache and getattr(self.model_cache[weights_path], "conf", None) == confidence:
            return self.model_cache[weights_path]
            
        print(f"Loading SAM3 Predictor from {weights_path}...")
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
            overrides = dict(conf=confidence, task="segment", mode="predict", model=weights_path, verbose=False)
            predictor = SAM3SemanticPredictor(overrides=overrides)
            self.model_cache[weights_path] = predictor
            setattr(predictor, "conf", confidence) # hack to track confidence state
            return predictor
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 model: {e}")

    def load_rfdetr(self, weights_path):
        if weights_path in self.model_cache:
            return self.model_cache[weights_path]
            
        if RFDETRMedium is None:
            raise ImportError("The 'rfdetr' library is not installed. Please install it to use RF-DETR models.")
            
        print(f"Loading RF-DETR from {weights_path}...")
        
        # Note: RF-DETR on MPS might be unstable as per user note.
        # device = "cpu" if self.device_str == "mps" else self.device_str
        
        # --- Dynamic Parameter Extraction ---
        def get_rf_metadata(checkpoint_path, default=1280):
            res = default
            class_names_dict = {}
            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if 'args' in ckpt:
                    args_obj = ckpt['args']
                    if hasattr(args_obj, 'resolution'):
                        res = args_obj.resolution
                    elif isinstance(args_obj, dict) and 'resolution' in args_obj:
                        res = args_obj['resolution']
                        
                    raw_classes = None
                    if hasattr(args_obj, 'class_names'):
                        raw_classes = args_obj.class_names
                    elif isinstance(args_obj, dict) and 'class_names' in args_obj:
                        raw_classes = args_obj['class_names']
                        
                    if isinstance(raw_classes, list):
                        class_names_dict = {i: name for i, name in enumerate(raw_classes)}
                    elif isinstance(raw_classes, dict):
                        class_names_dict = raw_classes
                        
                # Check for a custom JSON override next to the weights file
                import os, json
                custom_json_path = checkpoint_path.replace('.pth', '.json').replace('.pt', '.json')
                if os.path.exists(custom_json_path):
                    with open(custom_json_path, 'r') as f:
                        custom_classes = json.load(f)
                    
                    if isinstance(custom_classes, list):
                        # Support for a flat list or list of dicts (like COCO categories)
                        if len(custom_classes) > 0 and isinstance(custom_classes[0], dict) and 'name' in custom_classes[0]:
                            # Try to extract from COCO categories format and map to contiguous indices
                            names = [c['name'] for c in custom_classes]
                            class_names_dict = {i: name for i, name in enumerate(names)}
                        else:
                            class_names_dict = {i: name for i, name in enumerate(custom_classes)}
                    elif isinstance(custom_classes, dict):
                        # Extract string values sorted by integer keys if they are numbers
                        try:
                            sorted_names = [custom_classes[k] for k in sorted(custom_classes.keys(), key=int)]
                            class_names_dict = {i: name for i, name in enumerate(sorted_names)}
                        except ValueError:
                            class_names_dict = {i: name for i, name in enumerate(custom_classes.values())}

            except Exception as e:
                print(f"[!] Could not read metadata from checkpoint: {e}")
            return res, class_names_dict
            
        detected_res, detected_classes = get_rf_metadata(weights_path, default=1280)
        
        print(f"RF-DETR Config Detected -> Resolution: {detected_res}, Classes: {detected_classes}")

        # --- Dynamic Instantiation ---
        try:
            model = RFDETRMedium(
                pretrain_weights=weights_path,
                resolution=detected_res
            )
        except TypeError as te:
            if "'DetectionModel' object is not subscriptable" in str(te):
                raise RuntimeError(f"Failed to load RF-DETR model: The file '{os.path.basename(weights_path)}' appears to be a YOLO model (DetectionModel). Please select 'YOLO' model type.") from te
            raise te
        except Exception as e:
            # Re-raise as RuntimeError with context
             raise RuntimeError(f"Failed to load RF-DETR model: {e}") from e
        
        # Optimize for inference immediately
        try:
            model.optimize_for_inference()
        except Exception as opt_e:
            print(f"Warning: Failed to optimize RF-DETR for inference: {opt_e}. Continuing with standard model.")
            
        # Use a custom attribute because class_names is a read-only property
        model.custom_class_names = detected_classes
        
        self.model_cache[weights_path] = model
        return model

    def get_model_classes(self, model_type: str, model_path: str = None):
        """Returns list of class names or IDs. 
           (Kept for info purposes, though UI selection is removed)
        """
        if model_type.lower() in ["countgd", "sam2", "sam3", "groundingdino"]:
            return [] 
            
        if not model_path or not os.path.exists(model_path):
            return []
            
        try:
            if model_type.lower() == "yolo":
                model = self.load_yolo(model_path)
                return [{"id": k, "name": v} for k, v in model.names.items()]
                
            elif model_type.lower() == "rfdetr":
                model = self.load_rfdetr(model_path)
                classes = []
                # Check for custom_class_names first
                class_dict = getattr(model, 'custom_class_names', None)
                if not class_dict and hasattr(model, 'class_names') and model.class_names:
                    class_dict = model.class_names
                    
                if class_dict:
                    for cls_id, cls_name in class_dict.items():
                        classes.append({"id": cls_id, "name": cls_name})
                else:
                    classes.append({"id": 0, "name": "object"})
                return classes
                
        except Exception as e:
            print(f"Error fetching classes: {e}")
            return []
        
        return []

    def run_inference(self, image_path: str, model_type: str = "countgd", model_path: str = None, 
                      text_prompt: str = None, confidence: float = 0.25, selected_classes: list = None,
                      tiled: bool = False):
        
        img = Image.open(image_path).convert("RGB")
        w_img, h_img = img.size
        
        results = []
        
        if model_type.lower() == "countgd":
            self.load_countgd()
            
            if tiled and sv is not None:
                print(f"Running Tiled CountGD Inference on {image_path}...")
                
                def countgd_callback(image_slice: np.ndarray, model, transform, device, text_prompt, conf_thresh) -> sv.Detections:
                    slice_pil = Image.fromarray(cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB))
                    w_slice, h_slice = slice_pil.size
                    
                    boxes_norm = detector_logic.run_detector_inference(
                        model, transform, slice_pil, text_prompt, device, confidence_thresh=conf_thresh
                    )
                    
                    if not boxes_norm:
                        return sv.Detections.empty()
                        
                    xyxy_list = []
                    class_id_list = []
                    conf_list = []
                    
                    for b in boxes_norm:
                        # b is [class_id, xc, yc, w, h, conf] normalized
                        cid = b[0]
                        xc = b[1] * w_slice
                        yc = b[2] * h_slice
                        w = b[3] * w_slice
                        h = b[4] * h_slice
                        conf = b[5]
                        
                        x1 = xc - w / 2
                        y1 = yc - h / 2
                        x2 = xc + w / 2
                        y2 = yc + h / 2
                        
                        xyxy_list.append([x1, y1, x2, y2])
                        class_id_list.append(cid)
                        conf_list.append(conf)
                        
                    return sv.Detections(
                        xyxy=np.array(xyxy_list, dtype=np.float32),
                        confidence=np.array(conf_list, dtype=np.float32),
                        class_id=np.array(class_id_list, dtype=np.int32)
                    )

                callback_bound = partial(
                    countgd_callback, 
                    model=self.countgd_model, 
                    transform=self.countgd_transform, 
                    device=self.countgd_device, 
                    text_prompt=text_prompt, 
                    conf_thresh=confidence
                )
                
                countgd_slicer = sv.InferenceSlicer(
                    callback=callback_bound,
                    slice_wh=(640, 640),
                    iou_threshold=0.5
                )
                
                image_bgr = cv2.imread(image_path)
                detections = countgd_slicer(image_bgr)
                
                if hasattr(detections, 'xyxy'):
                    xyxy = detections.xyxy
                    class_ids = detections.class_id
                    confs = detections.confidence
                    
                    for i in range(len(xyxy)):
                        box = xyxy[i]
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        conf = float(confs[i]) if confs is not None else 1.0
                        label_to_use = text_prompt
                        
                        results.append({
                            "id": str(uuid.uuid4()),
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "label": label_to_use,
                            "confidence": conf
                        })
            else:
                # detector_logic returns: [class, xc, yc, w, h, conf] normalized
                boxes = detector_logic.run_detector_inference(
                    self.countgd_model, 
                    self.countgd_transform, 
                    img, 
                    text_prompt, 
                    self.countgd_device, 
                    confidence_thresh=confidence
                )
                
                for box in boxes:
                    # box = [class, xc, yc, w, h, conf]
                    xc, yc, w, h = box[1], box[2], box[3], box[4]
                    conf = box[5]
                    
                    # Convert normalized to pixels
                    width_px = w * w_img
                    height_px = h * h_img
                    x_px = (xc * w_img) - (width_px / 2)
                    y_px = (yc * h_img) - (height_px / 2)
                    
                    label_to_use = text_prompt

                    results.append({
                        "id": str(uuid.uuid4()),
                        "x": float(x_px),
                        "y": float(y_px),
                        "width": float(width_px),
                        "height": float(height_px),
                        "label": label_to_use,
                        "confidence": float(conf)
                    })



        elif model_type.lower() == "yolo":
            if not model_path:
                raise ValueError("Model path required for YOLO")

            if tiled and AutoDetectionModel is not None:
                # --- SAHI Tiled Inference ---
                print(f"Running Tiled YOLO Inference on {image_path}...")
                
                # Check for MPS
                device = self.device_str
                if device == "mps":
                     # SAHI might support mps if underlying ultralytics does, but let's be safe or just pass it
                     pass

                detection_model = AutoDetectionModel.from_pretrained(
                    model_type='yolov8',
                    model_path=model_path,
                    confidence_threshold=confidence,
                    device=device
                )

                result = get_sliced_prediction(
                    image_path,
                    detection_model,
                    slice_height=640,
                    slice_width=640,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2
                )

                # Convert SAHI results to our format
                for prediction in result.object_prediction_list:
                    # prediction.bbox is different? SAHI uses ShiftedBox
                    # bbox = [x_min, y_min, x_max, y_max]
                    bbox = prediction.bbox
                    x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                    
                    category = prediction.category
                    name = category.name
                    cid = category.id
                    score = prediction.score.value

                    if selected_classes and len(selected_classes) > 0 and cid not in selected_classes:
                        continue
                        
                    label = name
                    
                    results.append({
                        "id": str(uuid.uuid4()),
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "label": label,
                        "confidence": float(score)
                    })
            else:
                # --- Standard YOLO ---
                model = self.load_yolo(model_path)
                # YOLO inference
                res = model(img, device=self.device_str if self.device_str != "mps" else "mps", verbose=False, conf=confidence)[0]
                
                # Parse results
                names = model.names
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    if selected_classes and len(selected_classes) > 0 and cls_id not in selected_classes:
                        continue
                        
                    label = names[cls_id]
                    
                    # xyxy
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = coords
                    
                    results.append({
                        "id": str(uuid.uuid4()),
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "label": label,
                        "confidence": conf
                    })

        elif model_type.lower() == "groundingdino":
            if not text_prompt:
                raise ValueError("Text prompt required for GroundingDINO")
                
            self.load_grounding_dino()
            device_to_use = self.device_str
            
            if tiled and sv is not None:
                print(f"Running Tiled GroundingDINO Inference on {image_path}...")
                
                def dino_callback(image_slice: np.ndarray, processor, model, device, prompt, conf) -> sv.Detections:
                    slice_pil = Image.fromarray(cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB))
                    inputs = processor(images=slice_pil, text=prompt + ".", return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    target_sizes = torch.tensor([slice_pil.size[::-1]])
                    res_dino = processor.image_processor.post_process_object_detection(
                        outputs, threshold=conf, target_sizes=target_sizes
                    )[0]
                    
                    xyxy_list = []
                    conf_list = []
                    class_id_list = []
                    for score, label_id, box in zip(res_dino["scores"], res_dino["labels"], res_dino["boxes"]):
                        if float(score.item()) >= conf:
                            xyxy_list.append(box.tolist())
                            conf_list.append(float(score.item()))
                            class_id_list.append(0)
                            
                    if not xyxy_list:
                        return sv.Detections.empty()
                        
                    return sv.Detections(
                        xyxy=np.array(xyxy_list, dtype=np.float32),
                        confidence=np.array(conf_list, dtype=np.float32),
                        class_id=np.array(class_id_list, dtype=np.int32)
                    )

                callback_bound = partial(
                    dino_callback,
                    processor=self.grounding_dino_processor,
                    model=self.grounding_dino_model,
                    device=device_to_use,
                    prompt=text_prompt,
                    conf=confidence
                )
                
                slicer = sv.InferenceSlicer(callback=callback_bound, slice_wh=(640, 640), iou_threshold=0.5)
                image_bgr = cv2.imread(image_path)
                detections = slicer(image_bgr)
                
                if hasattr(detections, 'xyxy'):
                    for i in range(len(detections.xyxy)):
                        box = detections.xyxy[i]
                        c = float(detections.confidence[i]) if detections.confidence is not None else 1.0
                        results.append({
                            "id": str(uuid.uuid4()),
                            "x": float(box[0]),
                            "y": float(box[1]),
                            "width": float(box[2] - box[0]),
                            "height": float(box[3] - box[1]),
                            "label": text_prompt,
                            "confidence": c
                        })
            else:
                # Use original PIL image
                inputs = self.grounding_dino_processor(images=img, text=text_prompt + ".", return_tensors="pt").to(device_to_use)
                with torch.no_grad():
                    outputs = self.grounding_dino_model(**inputs)
                    
                target_sizes = torch.tensor([img.size[::-1]])
                res_dino = self.grounding_dino_processor.image_processor.post_process_object_detection(
                    outputs, threshold=confidence, target_sizes=target_sizes
                )[0]
                
                for score, label_id, box in zip(res_dino["scores"], res_dino["labels"], res_dino["boxes"]):
                    if float(score.item()) < confidence:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    results.append({
                        "id": str(uuid.uuid4()),
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "label": text_prompt,
                        "confidence": float(score.item())
                    })

        elif model_type.lower() == "sam2":
            if not model_path:
                raise ValueError("Model path required for SAM2")
            if not text_prompt:
                raise ValueError("Text prompt required for Grounded SAM2")
                
            # Step 1: Get initial bounding boxes using GroundingDINO (Text-to-box)
            initial_results = self.run_inference(
                image_path=image_path,
                model_type="groundingdino",
                text_prompt=text_prompt,
                confidence=confidence,
                tiled=tiled
            )
            
            if not initial_results:
                return []
                
            # Convert boxes to SAM format [x1, y1, x2, y2]
            input_boxes = []
            for res in initial_results:
                x1 = res["x"]
                y1 = res["y"]
                x2 = res["x"] + res["width"]
                y2 = res["y"] + res["height"]
                input_boxes.append([x1, y1, x2, y2])
                
            # Step 2: Pass boxes to SAM2 for refinement
            model = self.load_sam2(model_path)
            device_to_use = self.device_str
            # Actually Ultralytics SAM handles mps, we just pass self.device_str
            sam_res = model.predict(image_path, bboxes=input_boxes, device=device_to_use, verbose=False)[0]
            
            if hasattr(sam_res, 'boxes') and sam_res.boxes is not None and len(sam_res.boxes) > 0:
                for i, box in enumerate(sam_res.boxes):
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = coords
                    orig = initial_results[i] if i < len(initial_results) else {}
                    
                    calc_conf = float(box.conf[0].item()) if hasattr(box, 'conf') and box.conf is not None else float(orig.get("confidence", 1.0))
                    
                    results.append({
                        "id": str(uuid.uuid4()),
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "label": orig.get("label", text_prompt),
                        "confidence": calc_conf
                    })
            else:
                # Fallback to GroundingDINO results if SAM2 yields nothing (or encounters bad bounds)
                results = initial_results

        elif model_type.lower() == "sam3":
            if not model_path:
                raise ValueError("Model path required for SAM3")
            if not text_prompt:
                raise ValueError("Text prompt required for SAM3")
                
            predictor = self.load_sam3(model_path, confidence)
            
            if tiled and sv is not None:
                print(f"Running Tiled SAM3 Inference on {image_path}...")
                
                def sam3_callback(image_slice: np.ndarray, pred, prompt) -> sv.Detections:
                    pred.set_image(image_slice)
                    res_list = pred(text=[prompt])
                    
                    xyxy_list = []
                    conf_list = []
                    class_id_list = []
                    
                    if res_list and len(res_list) > 0:
                        sam_res = res_list[0]
                        if hasattr(sam_res, 'boxes') and sam_res.boxes is not None and len(sam_res.boxes) > 0:
                            for box in sam_res.boxes:
                                coords = box.xyxy[0].tolist()
                                conf = float(box.conf[0].item()) if hasattr(box, 'conf') and box.conf is not None else 1.0
                                xyxy_list.append(coords)
                                conf_list.append(conf)
                                class_id_list.append(0)
                    
                    if not xyxy_list:
                        return sv.Detections.empty()
                        
                    return sv.Detections(
                        xyxy=np.array(xyxy_list, dtype=np.float32),
                        confidence=np.array(conf_list, dtype=np.float32),
                        class_id=np.array(class_id_list, dtype=np.int32)
                    )

                callback_bound = partial(sam3_callback, pred=predictor, prompt=text_prompt)
                slicer = sv.InferenceSlicer(callback=callback_bound, slice_wh=(640, 640), iou_threshold=0.5)
                image_bgr = cv2.imread(image_path)
                detections = slicer(image_bgr)
                
                if hasattr(detections, 'xyxy'):
                    for i in range(len(detections.xyxy)):
                        box = detections.xyxy[i]
                        c = float(detections.confidence[i]) if detections.confidence is not None else 1.0
                        results.append({
                            "id": str(uuid.uuid4()),
                            "x": float(box[0]),
                            "y": float(box[1]),
                            "width": float(box[2] - box[0]),
                            "height": float(box[3] - box[1]),
                            "label": text_prompt,
                            "confidence": c
                        })
            else:
                predictor.set_image(image_path)
                res_list = predictor(text=[text_prompt])
                
                if res_list and len(res_list) > 0:
                    sam_res = res_list[0]
                    if hasattr(sam_res, 'boxes') and sam_res.boxes is not None and len(sam_res.boxes) > 0:
                        for box in sam_res.boxes:
                            coords = box.xyxy[0].tolist()
                            c = float(box.conf[0].item()) if hasattr(box, 'conf') and box.conf is not None else 1.0
                            x1, y1, x2, y2 = coords
                            results.append({
                                "id": str(uuid.uuid4()),
                                "x": float(x1),
                                "y": float(y1),
                                "width": float(x2 - x1),
                                "height": float(y2 - y1),
                                "label": text_prompt,
                                "confidence": c
                            })

        elif model_type.lower() == "rfdetr":
             if not model_path:
                raise ValueError("Model path required for RF-DETR")
                
             model = self.load_rfdetr(model_path)
             
             detections = None
             if tiled and sv is not None:
                 print(f"Running Tiled RF-DETR Inference on {image_path}...")
                 def rf_detr_callback(image_slice: np.ndarray, model) -> sv.Detections:
                    slice_pil = Image.fromarray(cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB))
                    raw_detections = model.predict(slice_pil, threshold=confidence)
                    return sv.Detections(
                        xyxy=raw_detections.xyxy,
                        confidence=raw_detections.confidence,
                        class_id=raw_detections.class_id
                    )

                 callback_with_model = partial(rf_detr_callback, model=model)
                 rf_slicer = sv.InferenceSlicer(
                    callback=callback_with_model,
                    slice_wh=(1280, 1280),
                    iou_threshold=0.5
                 )
                 # Slicer expects numpy BGR
                 image_bgr = cv2.imread(image_path)
                 detections = rf_slicer(image_bgr)
             else:
                 # Use a generic threshold or the one provided. eval3.py used 0.01 for mAP, but 0.25 is better for users.
                 detections = model.predict(img, threshold=confidence)
             
             # Map class IDs to names
             class_dict = getattr(model, 'custom_class_names', None)
             if not class_dict and hasattr(model, 'class_names') and model.class_names:
                 class_dict = model.class_names # {id: name}

             if hasattr(detections, 'xyxy'):
                 xyxy = detections.xyxy
                 class_ids = detections.class_id
                 confs = detections.confidence
                 
                 for i in range(len(xyxy)):
                     cid = int(class_ids[i]) if class_ids is not None else 0
                     conf = float(confs[i]) if confs is not None else 1.0
                     
                     if selected_classes and len(selected_classes) > 0 and cid not in selected_classes:
                         continue

                     box = xyxy[i]
                     x1, y1, x2, y2 = box
                     
                     if class_dict and cid in class_dict:
                         label_to_use = class_dict[cid]
                     else:
                         label_to_use = f"object_{cid}"

                     print(f"DEBUG: Detection {i}: CID={cid} -> Label={label_to_use}")

                     results.append({
                        "id": str(uuid.uuid4()),
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "label": label_to_use, 
                        "confidence": conf
                     })
             
        return results
