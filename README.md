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

Whenever you want to use the tool, run the start script. **We highly recommend using Google Chrome or Microsoft Edge** for the most stable experience, especially when dealing with massive datasets or file uploads.

### 🍎 Mac:
1. Double-click **`run_mac.command`**.
2. A terminal window will open, start the application server, and automatically open your web browser.
3. If it doesn't open automatically, or if you want to use a specific browser like Chrome, open it manually and navigate to `http://localhost:8000` or `http://127.0.0.1:8000`.
4. Keep the terminal window open while you use the app! To stop the tool, close the terminal window or press `Ctrl + C`.

### 🪟 Windows:
1. Double-click **`run_windows.bat`**.
2. A command prompt window will open to start the server, and your web browser will automatically open.
3. If it doesn't open automatically, or if you want to use a specific browser like Chrome, open it manually and navigate to `http://localhost:8000` or `http://127.0.0.1:8000`.
4. Keep the dark command prompt window open while you use the app! To stop the tool, freely close that window.

---

### 💡 Troubleshooting: "Network connection dropped" during model upload
If you are trying to upload a massive model file (e.g. `> 100MB`) and your browser continuously aborts the connection:
1. Open the video annotator folder in your file explorer (Finder/Explorer).
2. Manually copy your `.pt` or `.pth` weights file and paste it directly into the `data/models` subfolder.
3. Refresh the Annotator UI, and the model will instantly appear in the dropdown!

---

---

## 🎥 Demonstration
Here is a quick look at the Annotator Pro UI in action, demonstrating the CountGD text-prompt auto-annotation feature and the built-in Editor Tools:
[![Watch the video](https://img.youtube.com/vi/hS-LfiU4sec/maxresdefault.jpg)](https://youtu.be/hS-LfiU4sec)

---

## 🛠️ How It Works

Annotator Pro is designed to be purely local, fast, and feature-rich. Here is a detailed breakdown of each major section of the UI.

### 1. Media Import

The tool supports importing bulk image folders or extracting frames directly from video files.
- **Upload Video**: Click this to select an `.mp4` or `.mov` file. Specify your desired **Extract FPS** (Frames Per Second) and click **Extract**. The server will automatically break down the video into individual frames and load them.
- **Upload Images**: Click this to select a folder on your computer. If your images are stored in nested folders, Annotator Pro will automatically flatten the directory structure so all images appear in the main workspace.

When media is loaded, the files populate the **Images** panel on the left sidebar, acting as your directory tree. 

### 2. Auto-Annotation (AI)

Instead of manually drawing hundreds of boxes, use state-of-the-art AI models to automatically find objects for you. 

*   **Model Type**:
    *   **CountGD (Text Prompt)**: An open-set object detector. You do not need to train a model. Simply type the name of the object you want (e.g., "car", "person", "bird") in the **Prompt** box, and the AI will attempt to find it. Make sure you downloaded the weights during Setup!
    *   **YOLO (v8/v11)**: If you have your own trained model weights, upload your `.pt` file here. The system will extract the classes your model was trained on automatically.
    *   **RF-DETR**: Similar to YOLO, upload custom `.pt` weights for transformer-based detection.
*   **Confidence Slider**: Adjust the slider to filter out weak detections. Higher values (e.g. `0.65`) mean the AI has to be very sure, while lower values (e.g. `0.20`) will catch more objects but potentially produce more false positives.
*   **Enable Tiled Inference**: When hunting for very small objects in large high-resolution images, check this box. The AI will slice the image into smaller 640x640 overlapping tiles, run inference on each slice, and intelligently merge the boxes. Supported across all models!
*   **Override Label**: Sometimes a generic YOLO model only outputs "object". If you type "drone" into the override setting, it will force all boxes generated by the AI to be labeled "drone".
*   **Run on All Images**: Once you find the perfect confidence and prompt for your current image, you can click this to run the AI across your entire loaded dataset in the background.

### 3. Editor Tools & Manual Refinement

The main canvas is where you can manually draw, adjust, and correct annotations.
- **Draw Mode (D)**: Click and drag to draw new bounding boxes.
- **Pan Mode (P)**: Click and drag the image to move around. *Pro-tip: Hold the `Spacebar` while in Draw Mode to temporarily switch to Pan Mode!*
- **Zooming**: Use your mouse wheel to zoom in and out of the image smoothly. You can also use the `+` and `-` buttons in the sidebar.
- **Editing**: Click inside any existing box to select it. You can drag it to move it, or drag the corner handles to resize it perfectly.
- **Deleting**: Select a box and press the `Delete` or `Backspace` key on your keyboard to instantly remove it.
- **Labeling**: When a box is selected, you can change its class name in the "Label" text input located at the top toolbar.

### 4. Export & Verification

When you are done labeling your dataset:
- **Export to COCO**: Click this button to generate a downloadable `.zip` file containing all of your images and a standardized `_annotations.coco.json` file.
- **Verification Script**: If you want to double-check that your exported labels are perfectly aligned, run the included `verify_export_fix.py` script locally on the downloaded zip file to visually inspect the annotations.
