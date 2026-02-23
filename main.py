import os
import json
import shutil
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import time
import threading

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import video_processor
import detector_wrapper

import zipfile
import io
import uuid
import datetime
from PIL import Image

import webbrowser

# Data storage setup
DATA_DIR = os.path.join(os.getcwd(), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.json")

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)

# Startup: Clear data
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Clearing previous session data...")
    # Clear images
    if os.path.exists(IMAGES_DIR):
        for f in os.listdir(IMAGES_DIR):
            file_path = os.path.join(IMAGES_DIR, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    # Reset annotations
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump({}, f)
    print("Session data cleared.")
    
    yield

app = FastAPI(lifespan=lifespan)

# Auto-open browser
def open_browser():
    # Wait a bit for server to start
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

# Only run this if not in debug/reload mode
if os.environ.get("RUN_MAIN") != "true":
    threading.Thread(target=open_browser, daemon=True).start()

if not os.path.exists(ANNOTATIONS_FILE):
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump({}, f)

# Models
class Annotation(BaseModel):
    id: str  # Unique ID for the box
    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: Optional[float] = None

class ImageAnnotations(BaseModel):
    image_name: str
    boxes: List[Annotation]

MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class AutoAnnotateRequest(BaseModel):
    image_name: str
    text_prompt: Optional[str] = None
    confidence_thresh: float = 0.35
    model_type: str = "countgd" # countgd, yolo, rfdetr
    model_filename: Optional[str] = None
    selected_classes: Optional[List[int]] = None
    custom_label: Optional[str] = None # Override label name
    tiled: bool = False # Enable Tiled Inference (sahi/slicer)

@app.post("/api/auto_annotate")
async def auto_annotate(req: AutoAnnotateRequest):
    img_path = os.path.join(IMAGES_DIR, req.image_name)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    model_path = None
    if req.model_filename:
        model_path = os.path.join(MODEL_DIR, req.model_filename)
        
    try:
        detector = detector_wrapper.DetectorWrapper.get_instance()
        new_boxes = detector.run_inference(
            image_path=img_path, 
            model_type=req.model_type,
            model_path=model_path,
            text_prompt=req.text_prompt, 
            confidence=req.confidence_thresh,
            selected_classes=req.selected_classes,
            custom_label=req.custom_label,
            tiled=req.tiled
        )
        
        return {"boxes": new_boxes, "count": len(new_boxes)}
    except Exception as e:
        print(f"Auto-annotation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload_model")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".pt") and not file.filename.endswith(".pth"):
         raise HTTPException(status_code=400, detail="Only .pt or .pth files supported")
         
    file_path = os.path.join(MODEL_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "message": "Model uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_models():
    """List available model files"""
    models = []
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith(".pt") or f.endswith(".pth"):
                models.append(f)
    return {"models": models}

class ModelClassesRequest(BaseModel):
    model_type: str
    model_filename: str

@app.post("/api/load_model_classes")
async def load_model_classes(req: ModelClassesRequest):
    model_path = os.path.join(MODEL_DIR, req.model_filename)
    if not os.path.exists(model_path):
         raise HTTPException(status_code=404, detail="Model file not found")
         
    try:
        detector = detector_wrapper.DetectorWrapper.get_instance()
        classes = detector.get_model_classes(req.model_type, model_path)
        return {"classes": classes}
    except Exception as e:
        print(f"Error loading classes: {e}")
        return {"classes": [], "error": str(e)}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...), fps: float = Form(1.0)):
    temp_path = os.path.join(DATA_DIR, "temp_video.mp4")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Clear existing images for a fresh start? 
        # For this simple tool, let's clear previous images when a new video is uploaded
        for f in os.listdir(IMAGES_DIR):
            os.remove(os.path.join(IMAGES_DIR, f))
            
        # Use filename as prefix
        prefix = "frame"
        if file.filename:
            # Sanitize: remove extension and weird chars
            clean_name = os.path.splitext(file.filename)[0]
            clean_name = "".join([c if c.isalnum() else "_" for c in clean_name])
            if clean_name:
                prefix = clean_name
            
        count = video_processor.extract_frames(temp_path, IMAGES_DIR, fps, prefix=prefix)
        
        # Reset annotations
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump({}, f)
            
        return {"message": f"Extracted {count} frames", "count": count}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/upload_images")
async def upload_images_folder(files: List[UploadFile] = File(...), clear_existing: bool = True):
    # Clear existing images only if requested
    if clear_existing:
        print("Clearing existing images...")
        for f in os.listdir(IMAGES_DIR):
            os.remove(os.path.join(IMAGES_DIR, f))
        
    count = 0
    for file in files:
        if file.filename:
            # Flatten path (ignore folder structure)
            filename = os.path.basename(file.filename)
            # Skip hidden files like .DS_Store
            if filename.startswith('.'):
                continue
                
            path = os.path.join(IMAGES_DIR, filename)
            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            count += 1
    
    # Reset annotations
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump({}, f)
        
    return {"message": f"Uploaded {count} images", "count": count}

@app.get("/api/images")
async def get_images():
    images = video_processor.list_images(IMAGES_DIR)
    return {"images": images}

@app.get("/api/annotations/{image_name}")
async def get_annotations(image_name: str):
    with open(ANNOTATIONS_FILE, 'r') as f:
        data = json.load(f)
    return data.get(image_name, [])

@app.post("/api/annotations")
async def save_annotations(data: ImageAnnotations):
    print(f"DEBUG: Saving annotations for {data.image_name}. Count: {len(data.boxes)}")
    for b in data.boxes:
        print(f" - Box ID: {b.id}, Label: {b.label}")

    with open(ANNOTATIONS_FILE, 'r') as f:
        all_data = json.load(f)
    
    all_data[data.image_name] = [box.dict() for box in data.boxes]
    
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(all_data, f)
    return {"status": "success"}

@app.post("/api/reset_dataset")
async def reset_dataset():
    """Clears all images and annotations to start fresh."""
    try:
        # Clear images
        for filename in os.listdir(IMAGES_DIR):
            file_path = os.path.join(IMAGES_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Reset annotations
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump({}, f)
            
        return {"status": "success", "message": "Dataset reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Basic in-memory job store
export_jobs = {}

def run_export_task(job_id: str):
    """Background task to generate COCO zip"""
    try:
        job = export_jobs[job_id]
        job["status"] = "processing"
        
        with open(ANNOTATIONS_FILE, 'r') as f:
            all_annotations = json.load(f)
        
        coco = {
            "info": {
                "year": str(datetime.datetime.now().year),
                "version": "1.0",
                "description": "Exported Dataset",
                "contributor": "",
                "url": "",
                "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                    "name": "CC BY 4.0"
                }
            ],
            "categories": [],
            "images": [],
            "annotations": []
        }
        
        # Pre-seed category map to match specific model requirements
        # User defined: 0=person, 1=animal
        category_map = {"person": 0, "animal": 1}
        next_cat_id = 2
        
        image_id = 0 
        ann_id = 0
        
        images_list = video_processor.list_images(IMAGES_DIR)
        job["total"] = len(images_list)
        
        valid_images = []
        
        processed_count = 0

        for img_name in images_list:
            # Update progress periodically
            processed_count += 1
            if processed_count % 10 == 0:
                job["current"] = processed_count
                job["message"] = f"Processing image {processed_count}/{job['total']}"
            
            img_path = os.path.join(IMAGES_DIR, img_name)
            if not os.path.exists(img_path):
                continue
                
            # Use PIL for lazy loading of dimensions (much faster than cv2.imread)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                continue

            # File date
            mtime = os.path.getmtime(img_path)
            dt = datetime.datetime.fromtimestamp(mtime)
            date_captured = dt.strftime('%Y-%m-%d %H:%M:%S')

            valid_images.append(img_path)

            coco["images"].append({
                "id": image_id,
                "license": 1,
                "file_name": img_name,
                "height": h,
                "width": w,
                "date_captured": date_captured,
                "extra": {
                    "name": img_name
                }
            })
            
            # Add annotations
            if img_name in all_annotations:
                for box in all_annotations[img_name]:
                    label = box.get("label", "object")
                    if label not in category_map:
                        category_map[label] = next_cat_id
                        next_cat_id += 1
                    
                    cat_id = category_map[label]
                    
                    # COCO box format: [x, y, width, height]
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [box["x"], box["y"], box["width"], box["height"]],
                        "area": box["width"] * box["height"],
                        "iscrowd": 0,
                        "segmentation": [] 
                    })
                    ann_id += 1
            
            image_id += 1
            
        # Update categories in COCO
        coco_categories = []
        for name, cid in category_map.items():
            coco_categories.append({
                "id": cid, 
                "name": name, 
                "supercategory": "Object"
            })
        
        # If empty
        if not coco_categories:
             pass 
            
        coco["categories"] = coco_categories
        
        # Create Zip
        job["message"] = "Compressing archive..."
        zip_filename = f"export_{job_id}.zip"
        zip_path = os.path.join(DATA_DIR, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            json_str = json.dumps(coco, indent=4)
            zipf.writestr("_annotations.coco.json", json_str)
            for img_path in valid_images:
                zipf.write(img_path, arcname=os.path.join("images", os.path.basename(img_path)))
                
        job["file_path"] = zip_path
        job["status"] = "completed"
        job["current"] = job["total"]
        job["message"] = "Done!"
        
    except Exception as e:
        print(f"Export Job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)


@app.post("/api/export/start")
async def start_export():
    job_id = str(uuid.uuid4())
    export_jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "total": 0,
        "current": 0,
        "message": "Starting...",
        "file_path": None,
        "error": None
    }
    
    thread = threading.Thread(target=run_export_task, args=(job_id,))
    thread.start()
    
    return {"job_id": job_id}

@app.get("/api/export/status/{job_id}")
async def get_export_status(job_id: str):
    if job_id not in export_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return export_jobs[job_id]

def cleanup_after_export(zip_path: str):
    """Deletes the export zip and clears the dataset."""
    try:
        # 1. Delete the zip file - DISABLED to allow recovery
        # if os.path.exists(zip_path):
        #     os.remove(zip_path)
            
        # 2. Clear images
        for filename in os.listdir(IMAGES_DIR):
            file_path = os.path.join(IMAGES_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        # 3. Reset annotations
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump({}, f)
            
        print(f"Cleanup complete for {zip_path}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

@app.get("/api/export/download/{job_id}")
async def download_export(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in export_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = export_jobs[job_id]
    if job["status"] != "completed" or not job["file_path"] or not os.path.exists(job["file_path"]):
        raise HTTPException(status_code=404, detail="Export not ready or file missing")
    
    zip_path = job["file_path"]
    
    # Schedule cleanup to run AFTER the response is sent
    background_tasks.add_task(cleanup_after_export, zip_path)
    
    return FileResponse(zip_path, filename="dataset_export.zip", media_type="application/zip")

# Mount static files - MUST be last to avoid overriding API
# Mount images first so it is not caught by the root static mount
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# Determine path to static files
import sys

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Check if running as a script or frozen onedir
        if getattr(sys, 'frozen', False):
             # If onedir, resources are likely in the same dir as the executable
             base_path = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

static_path = get_resource_path("static")
print(f"DEBUG: Static path resolved to: {static_path}")

if not os.path.exists(static_path):
     # Fallback: Try looking one level up? Or just fail with clear message
     print(f"WARNING: Static dir not found at {static_path}. Trying current directory.")
     static_path = os.path.abspath("static")

if not os.path.exists(static_path):
    raise RuntimeError(f"Directory '{static_path}' does not exist. Current CWD: {os.getcwd()}")

app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    print("Starting AnnotatorV2 server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
