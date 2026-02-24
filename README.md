# Annotator Pro

A powerful, web-based tool for annotating images and video frames for computer vision tasks. It supports manual annotation and AI-powered auto-annotation using state-of-the-art models like CountGD, YOLO, and RF-DETR.

## Prerequisites: Install Python

Before you can run Annotator Pro, your computer needs to have Python installed.

### 🍎 For Mac Users:
1. Go to the [Python Downloads page for Mac](https://www.python.org/downloads/mac-osx/).
2. Download the latest "macOS 64-bit universal2 installer" (Python 3.9 or newer).
3. Open the downloaded file and click through the installer using the default settings.
4. *(Alternative for advanced users)*: Install via Homebrew in the terminal: `brew install python3`

### 🪟 For Windows Users:
1. Go to the [Python Downloads page for Windows](https://www.python.org/downloads/windows/).
2. Download the "Windows installer (64-bit)" (Python 3.9 or newer).
3. Open the downloaded installer.
4. **CRITICAL STEP**: At the very bottom of the first setup screen, **check the box that says "Add Python.exe to PATH"**. If you miss this step, the setup scripts will not work.
5. Click "Install Now".

---

## Installation & First-Time Setup

Once Python is installed, follow these instructions to set up the tool. You only need to do this **once**.

### 0. Download CountGD Weights
If you plan to use the "CountGD (Text Prompt)" AI auto-annotation feature, you must first download the model weights:
1. Download the `checkpoint_fsc147_best.pth` file from this [GitHub Release](https://github.com/ChaimElchik/Annotator-Pro/releases/download/v1.0.0/checkpoint_fsc147_best.pth).
2. Place the downloaded `.pth` file directly into your `Annotator-Pro` (or `Github version`) folder (the same folder that contains `main.py`).

### 🍎 Mac:
1. Open the folder containing the Video Annotator Pro files.
2. Double-click **`setup_mac.command`**.
3. A terminal window will open and automatically download everything needed. This may take a few minutes.
4. When it says "Setup Complete!", close the terminal window.

### 🪟 Windows:
1. Open the folder containing the Video Annotator Pro files.
2. Double-click **`setup_windows.bat`**.
3. A command prompt window will open and automatically download everything needed. This may take a few minutes.
4. When it says "Setup Complete!", close the window.

---

## Starting the Application

Whenever you want to use the tool, run the start script. 

### 🍎 Mac:
1. Double-click **`run_mac.command`**.
2. A terminal window will open, start the application server, and automatically open your web browser to `http://localhost:8000`.
3. Keep the terminal window open while you use the app! To stop the tool, close the terminal window or press `Ctrl + C`.

### 🪟 Windows:
1. Double-click **`run_windows.bat`**.
2. A command prompt window will open to start the server, and your web browser will automatically open to `http://localhost:8000`.
3. Keep the dark command prompt window open while you use the app! To stop the tool, freely close that window.

---

## Features

### 1. Media Import
- **Upload Video**: Upload a video file to automatically extract frames at a customizable FPS (Frames Per Second).
- **Upload Images**: Upload a folder of images. Nested folders are automatically flattened so all images are imported into the main workspace.

### 2. Manual Annotation
- **Draw Mode**: Click and drag to draw bounding boxes.
- **Pan/Zoom**: Use the Pan tool or hold Spacebar + Click/Drag to navigate large images. Mouse wheel to zoom in/out.
- **Editing**: Select existing boxes to move or resize (using corner handles).
- **Deleting**: Select a box and press `Delete` or `Backspace` (or use the UI button) to remove it.

### 3. AI Auto-Annotation
Accelerate your workflow by using AI models to detect objects automatically.

- **Models Supported**:
    - **CountGD**: Open-set detector. Enter a **Text Prompt** (e.g., "person", "car") to detect specific objects without custom training.
    - **YOLO (v8/v11)**: Upload your custom `.pt` weights file. The system will use your trained model for inference.
    - **RF-DETR**: Upload custom `.pt` weights for transformer-based detection.

- **Smart Features**:
    - **Custom Label Override**: Force all detections to have a specific label (e.g., if your generic model detects "object" but you want "drone").
    - **Confidence Threshold**: Adjust the slider to filter out weak detections.
    - **Run on All**: Batch process the entirely loaded dataset matching your current settings.
    - **Model Caching**: Loaded models are cached in memory for faster subsequent runs.

### 4. Export & Verification
- **Export to COCO**: Downloads a `.zip` file containing:
    - All images.
    - `_annotations.coco.json`: Standard COCO format annotations.
- **Verification Script**: Use the included `verify_export_fix.py` to visually verify the exported dataset.

---

## Workflow Example
1.  **Import**: Click "Upload Video" (and set FPS) or "Upload Images" to load your dataset.
2.  **Auto-Annotate (Optional)**:
    -   Select a Model Type.
    -   (For YOLO/RF-DETR) Upload your model `.pt` weights file using the UI.
    -   (Optional) Enter an "Override Label" name.
    -   Click "Run Auto-Annotation" for the current image or "Run on All Images" for the whole set.
3.  **Refine**: Use the manual tools to fix mistakes, adjust boxes, or add missing labels.
4.  **Export**: Click "Export to COCO" to download your dataset.
