const state = {
    images: [],
    currentImageIndex: -1,
    annotations: {}, // Map of image_name -> [boxes]
    isDrawing: false,
    startX: 0,
    startY: 0,
    currentLabel: 'object',
    imageObj: null, // The current Image object

    // The View System (Transform based)
    view: {
        x: 0,
        y: 0,
        scale: 1.0
    },

    // Interaction
    selectedBoxId: null,
    dragMode: null, // 'move', 'resize-tl', 'resize-tr', 'resize-bl', 'resize-br', 'create'
    dragStartBox: null, // Copy of box at start of drag
    dragStartX: 0,
    dragStartY: 0,

    // Pan state
    toolMode: 'draw', // 'draw' | 'pan'
    isPanning: false,
    panStartX: 0,
    panStartY: 0,
    viewStartX: 0, // Store view.x at start of pan
    viewStartY: 0  // Store view.y at start of pan
};

// DOM Elements
const els = {
    uploadVideoBtn: document.getElementById('btn-upload-video'),
    videoInput: document.getElementById('input-video'),
    uploadImagesBtn: document.getElementById('btn-upload-images'),
    imagesInput: document.getElementById('input-images'),
    fpsControl: document.getElementById('fps-control'),
    fpsInput: document.getElementById('fps-input'),
    extractBtn: document.getElementById('btn-extract'),
    imageList: document.getElementById('image-list'),
    imageCount: document.getElementById('image-count'),
    canvas: document.getElementById('image-canvas'),
    canvasWrapper: document.getElementById('canvas-wrapper'),
    // canvasContainer is now the object getting transformed
    canvasContainer: document.querySelector('.canvas-container'),
    emptyState: document.getElementById('empty-state'),
    filenameDisplay: document.getElementById('current-filename'),
    prevBtn: document.getElementById('btn-prev'),
    nextBtn: document.getElementById('btn-next'),
    counter: document.getElementById('image-counter'),
    labelInput: document.getElementById('current-label'),
    clearBtn: document.getElementById('btn-clear'),
    exportBtn: document.getElementById('btn-export'),

    // Auto Annotate
    aaModelType: document.getElementById('aa-model-type'),
    aaTiled: document.getElementById('aa-tiled'), // New
    sectionCountGD: document.getElementById('section-countgd'),
    sectionCustomModel: document.getElementById('section-custom-model'),

    aaPrompt: document.getElementById('aa-prompt'),
    aaCustomLabel: document.getElementById('aa-custom-label'),

    aaModelFile: document.getElementById('aa-model-file'),
    btnUploadModel: document.getElementById('btn-upload-model'),
    inputModelFile: document.getElementById('input-model-file'),

    // Classes Display
    modelClassesDisplay: document.getElementById('model-classes-display'),
    modelClassesList: document.getElementById('model-classes-list'),

    aaConf: document.getElementById('aa-conf'),
    aaConfVal: document.getElementById('conf-val'),
    aaBtn: document.getElementById('btn-auto-annotate'),
    aaBtnAll: document.getElementById('btn-auto-annotate-all'),
    aaStatus: document.getElementById('aa-status'),

    // Edior Tools
    modeDrawBtn: document.getElementById('btn-mode-draw'),
    modePanBtn: document.getElementById('btn-mode-pan'),
    zoomInBtn: document.getElementById('btn-zoom-in'),
    zoomOutBtn: document.getElementById('btn-zoom-out'),
    zoomResetBtn: document.getElementById('btn-zoom-reset'),
    zoomDisplay: document.getElementById('zoom-level'),
    deleteSelectedBtn: document.getElementById('btn-delete-selected'),

    // Export Modal
    exportModal: document.getElementById('export-modal'),
    exportProgressBar: document.getElementById('export-progress-bar'),
    exportStatusText: document.getElementById('export-status-text'),
    exportActions: document.getElementById('export-actions'),
    btnModalReset: document.getElementById('btn-modal-reset'),
    btnModalClose: document.getElementById('btn-modal-close'),

    // Reset
    btnResetDataset: document.getElementById('btn-reset-dataset')
};

const ctx = els.canvas.getContext('2d');

// --- Initialization ---

function logToDiv(msg) {
    console.log(msg);
}

function init() {
    setupEventListeners();
    fetchImageList();
    fetchModels();
}

function setupEventListeners() {
    // Uploads
    els.uploadVideoBtn.addEventListener('click', () => els.videoInput.click());
    els.videoInput.addEventListener('change', handleVideoSelect);

    els.uploadImagesBtn.addEventListener('click', () => els.imagesInput.click());
    els.imagesInput.addEventListener('change', handleImagesUpload);

    // FPS Extract (for video flow)
    els.extractBtn.addEventListener('click', handleExtractFrames);

    // Auto Annotate
    els.aaConf.addEventListener('input', (e) => els.aaConfVal.innerText = e.target.value);

    els.aaModelType.addEventListener('change', handleModelTypeChange);
    els.btnUploadModel.addEventListener('click', () => els.inputModelFile.click());
    els.inputModelFile.addEventListener('change', handleModelUpload);

    // Load classes when model is selected
    els.aaModelFile.addEventListener('change', handleModelSelectionChange);

    els.aaBtn.addEventListener('click', handleAutoAnnotate);
    els.aaBtnAll.addEventListener('click', handleAutoAnnotateAll);

    // Navigation
    els.prevBtn.addEventListener('click', () => { loadImage(state.currentImageIndex - 1); });
    els.nextBtn.addEventListener('click', () => { loadImage(state.currentImageIndex + 1); });

    // Canvas interactions
    // Mouse Down must be on canvas to start
    els.canvas.addEventListener('mousedown', handleMouseDown);

    // Move and Up should be global to handle dragging outside canvas
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // Tools
    els.clearBtn.addEventListener('click', clearAnnotations);
    els.exportBtn.addEventListener('click', exportData);

    // Reset Controls
    els.btnResetDataset.addEventListener('click', handleResetDataset);
    els.btnModalReset.addEventListener('click', () => {
        els.exportModal.classList.add('hidden');
        handleResetDataset();
    });
    els.btnModalClose.addEventListener('click', () => els.exportModal.classList.add('hidden'));

    els.deleteSelectedBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        deleteSelectedBox();
    });

    els.modeDrawBtn.addEventListener('click', () => setToolMode('draw'));
    els.modePanBtn.addEventListener('click', () => setToolMode('pan'));

    // Zoom via Wheel
    els.canvasWrapper.addEventListener('wheel', handleWheel, { passive: false });

    // Buttons (Keep as accessible alternative)
    els.zoomInBtn.addEventListener('click', () => updateZoomStep(0.2));
    els.zoomOutBtn.addEventListener('click', () => updateZoomStep(-0.2));
    els.zoomResetBtn.addEventListener('click', resetZoom);

    els.labelInput.addEventListener('change', (e) => {
        state.currentLabel = e.target.value;
        if (state.selectedBoxId) {
            const imageName = state.images[state.currentImageIndex];
            const box = state.annotations[imageName]?.find(b => String(b.id) === String(state.selectedBoxId)); // Robust string check
            if (box) {
                box.label = state.currentLabel;
                saveAnnotations(imageName);
                redraw();
            }
        }
    });

    // Keyboard nav
    window.addEventListener('keydown', (e) => {
        // Debug key
        // logToDiv(`Key: ${e.key} (${e.code})`);

        if (e.target.tagName === 'INPUT') return;

        // Force Delete catch
        if (e.key === 'Delete' || e.key === 'Backspace') {
            e.preventDefault();
            e.stopPropagation();
            logToDiv(`Global KeyDown: ${e.key} - Deleting...`);
            deleteSelectedBox();
            return;
        }

        if (e.key === 'ArrowLeft') loadImage(state.currentImageIndex - 1);
        if (e.key === 'ArrowRight') loadImage(state.currentImageIndex + 1);

        // Spacebar Pan Toggle (Quick Mode)
        if (e.code === 'Space' && !e.repeat) {
            e.preventDefault(); // prevent scroll down
            if (state.toolMode !== 'pan') {
                state.prevToolMode = state.toolMode;
                setToolMode('pan');
            }
        }
    }, { capture: true }); // Important! Capture phase

    document.addEventListener('keyup', (e) => {
        if (e.code === 'Space') {
            if (state.prevToolMode) {
                setToolMode(state.prevToolMode);
                state.prevToolMode = null;
            }
        }
    });
}

// --- API Calls ---

async function fetchImageList() {
    try {
        const res = await fetch('/api/images');
        const data = await res.json();
        state.images = data.images;
        updateImageListUI();

        if (state.images.length > 0) {
            loadImage(0);
        } else {
            showEmptyState();
        }
    } catch (err) {
        console.error("Failed to fetch images", err);
    }
}

async function fetchModels() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        const models = data.models;

        els.aaModelFile.innerHTML = '<option value="" disabled selected>Select a model file...</option>';
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.innerText = m;
            els.aaModelFile.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to fetch models", err);
    }
}

async function fetchAnnotations(imageName) {
    try {
        const res = await fetch(`/api/annotations/${imageName}`);
        const data = await res.json();
        // Check if we already have local changes for this image?
        // For simplicity, always trust server on load,
        // but since we save immediately on change, it should be fine.
        state.annotations[imageName] = data || [];
    } catch (err) {
        console.error("Failed to fetch annotations", err);
        state.annotations[imageName] = [];
    }
}

async function saveAnnotations(imageName) {
    const boxes = state.annotations[imageName];
    try {
        await fetch('/api/annotations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_name: imageName,
                boxes: boxes
            })
        });
    } catch (err) {
        console.error("Failed to save", err);
    }
}

// --- Handlers ---

function handleModelTypeChange() {
    const type = els.aaModelType.value;

    // Clear potentially stale classes first
    displayModelClasses([]);

    // Reset upload button state
    els.btnUploadModel.innerText = "Upload Weights (.pt)";
    els.btnUploadModel.disabled = false;

    if (type === 'countgd') {
        els.sectionCountGD.classList.remove('hidden');
        els.sectionCustomModel.classList.add('hidden');
    } else {
        els.sectionCountGD.classList.add('hidden');
        els.sectionCustomModel.classList.remove('hidden');

        // Reset the selected model file so users must explicitly choose one for the new type
        els.aaModelFile.value = "";
    }
}

async function handleModelUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Check extension
    if (!file.name.endsWith('.pt') && !file.name.endsWith('.pth')) {
        alert("Please select a .pt or .pth file.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    els.btnUploadModel.innerText = "Uploading...";
    els.btnUploadModel.disabled = true;

    try {
        const res = await fetch('/api/upload_model', {
            method: 'POST',
            body: formData
        });
        if (!res.ok) throw new Error("Upload failed");

        await fetchModels();
        // Select the uploaded model
        els.aaModelFile.value = file.name;
        alert("Model weights uploaded successfully!");

    } catch (err) {
        console.error(err);
        alert("Model upload failed: " + err.message);
    } finally {
        els.inputModelFile.value = '';
        els.btnUploadModel.innerText = "Upload Weights (.pt)";
        els.btnUploadModel.disabled = false;
    }
}


async function handleModelSelectionChange() {
    const filename = els.aaModelFile.value;
    const type = els.aaModelType.value;
    if (!filename || !type) return;

    // Clear custom label to encourage native class usage
    els.aaCustomLabel.value = "";

    try {
        const res = await fetch('/api/load_model_classes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_type: type, model_filename: filename })
        });
        const data = await res.json();

        const classes = data.classes || [];
        displayModelClasses(classes);

    } catch (err) {
        console.error("Failed to load classes", err);
        displayModelClasses([]);
    }
}

function displayModelClasses(classes) {
    els.modelClassesList.innerHTML = '';
    if (classes.length === 0) {
        els.modelClassesDisplay.classList.add('hidden');
        return;
    }

    els.modelClassesDisplay.classList.remove('hidden');
    classes.forEach(cls => {
        const span = document.createElement('span');
        span.className = 'model-class-tag';
        // Handle {id, name} object or string
        const name = cls.name || cls;
        span.innerText = name;
        els.modelClassesList.appendChild(span);
    });
}



function handleVideoSelect(e) {
    if (e.target.files.length === 0) return;
    // Show FPS control, hide images?
    els.fpsControl.classList.remove('hidden');
    // We don't upload yet, we wait for 'Extract' click
}

async function handleExtractFrames() {
    const file = els.videoInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('fps', els.fpsInput.value);

    els.extractBtn.innerText = "Processing...";
    els.extractBtn.disabled = true;

    try {
        const res = await fetch('/api/upload_video', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        console.log("Extracted", data);
        await fetchImageList();
        els.fpsControl.classList.add('hidden'); // Hide after done
    } catch (err) {
        alert("Error extracting frames");
        console.error(err);
    } finally {
        els.extractBtn.innerText = "Extract";
        els.extractBtn.disabled = false;
        els.videoInput.value = ''; // Reset
    }
}

async function handleImagesUpload(e) {
    const files = els.imagesInput.files;
    if (files.length === 0) return;

    const BATCH_SIZE = 100;
    const totalFiles = files.length;
    const totalBatches = Math.ceil(totalFiles / BATCH_SIZE);

    els.uploadImagesBtn.disabled = true;

    try {
        for (let i = 0; i < totalBatches; i++) {
            const start = i * BATCH_SIZE;
            const end = Math.min(start + BATCH_SIZE, totalFiles);
            const batchFiles = [];

            // Create FormData for this batch
            const formData = new FormData();
            for (let j = start; j < end; j++) {
                formData.append('files', files[j]);
            }

            els.uploadImagesBtn.innerText = `Uploading ${i + 1}/${totalBatches}...`;

            // Clear existing only on the first batch
            const clearExisting = (i === 0);

            const res = await fetch(`/api/upload_images?clear_existing=${clearExisting}`, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error(`Batch ${i + 1} failed`);
        }

        await fetchImageList();
        alert(`Successfully uploaded ${totalFiles} images.`);

    } catch (err) {
        console.error(err);
        alert("Upload failed: " + err.message);
    } finally {
        els.uploadImagesBtn.innerText = "Upload Images";
        els.uploadImagesBtn.disabled = false;
        els.imagesInput.value = '';
    }
}

async function handleAutoAnnotate() {
    if (state.currentImageIndex < 0) return;
    const imageName = state.images[state.currentImageIndex];

    const type = els.aaModelType.value;
    let prompt = null;
    let filename = null;

    if (type === 'countgd') {
        prompt = els.aaPrompt.value.trim();
        if (!prompt) {
            alert("Please enter a text prompt");
            return;
        }
    } else {
        filename = els.aaModelFile.value;
        if (!filename) {
            alert("Please select a model file");
            return;
        }
        prompt = "detected_object";
    }

    els.aaBtn.disabled = true;
    els.aaBtn.innerText = "Running...";
    els.aaStatus.innerText = "Loading model & inferencing...";

    try {
        const payload = {
            image_name: imageName,
            confidence_thresh: parseFloat(els.aaConf.value),
            model_type: type,
            tiled: els.aaTiled.checked
        };

        if (prompt) payload.text_prompt = prompt;
        if (filename) payload.model_filename = filename;

        const customLabel = els.aaCustomLabel.value.trim();
        if (customLabel) payload.custom_label = customLabel;

        const res = await fetch('/api/auto_annotate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || res.statusText);
        }

        const data = await res.json();
        const newBoxes = data.boxes;

        if (!state.annotations[imageName]) state.annotations[imageName] = [];
        state.annotations[imageName].push(...newBoxes);

        await saveAnnotations(imageName);
        redraw();

        const counts = {};
        newBoxes.forEach(b => {
            counts[b.label] = (counts[b.label] || 0) + 1;
        });
        const summary = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(', ');

        els.aaStatus.innerText = summary ? `Found: ${summary}` : "No objects found.";
        setTimeout(() => els.aaStatus.innerText = "", 5000);

    } catch (err) {
        console.error(err);
        els.aaStatus.innerText = "Error during annotation.";
        alert("Auto-annotation failed: " + err.message);
    } finally {
        els.aaBtn.disabled = false;
        els.aaBtn.innerText = "Run Auto-Annotation";
    }
}

// --- UI Logic ---

function showEmptyState() {
    els.emptyState.style.display = 'block';
    els.canvasContainer.style.display = 'none';
    els.filenameDisplay.innerText = "No image selected";
}

async function loadImage(index) {
    if (index < 0 || index >= state.images.length) return;

    state.currentImageIndex = index;
    const imageName = state.images[index];

    // Update UI highlights
    document.querySelectorAll('.image-list li').forEach((li, i) => {
        if (i === index) li.classList.add('active');
        else li.classList.remove('active');
    });

    els.emptyState.style.display = 'none';
    els.canvasContainer.style.display = 'block';
    els.filenameDisplay.innerText = imageName;
    els.counter.innerText = `${index + 1} / ${state.images.length}`;

    // Fetch Annotations
    await fetchAnnotations(imageName);

    // Load Image onto Canvas
    const img = new Image();
    img.src = `/images/${imageName}`; // Served by FastAPI
    img.onload = () => {
        state.imageObj = img;
        // Set canvas dimensions to natural image size
        els.canvas.width = img.naturalWidth;
        els.canvas.height = img.naturalHeight;
        fitImageToScreen(); // Adjust view to fit
    };
}

function fitImageToScreen() {
    if (!state.imageObj) return;

    const wrapperW = els.canvasWrapper.clientWidth;
    const wrapperH = els.canvasWrapper.clientHeight;
    const imgW = state.imageObj.naturalWidth;
    const imgH = state.imageObj.naturalHeight;

    // Calculate scale to fit image within wrapper, with some padding
    const scale = Math.min((wrapperW - 40) / imgW, (wrapperH - 40) / imgH, 1);

    // Center the image
    const x = (wrapperW - imgW * scale) / 2;
    const y = (wrapperH - imgH * scale) / 2;

    state.view = { x, y, scale };
    updateTransform();
    redraw();
}

function updateTransform() {
    els.canvasContainer.style.transform = `translate(${state.view.x}px, ${state.view.y}px) scale(${state.view.scale})`;
    els.zoomDisplay.innerText = Math.round(state.view.scale * 100) + "%";
}

function getMousePos(e) {
    // Convert screen coordinates to image coordinates (canvas drawing coordinates)
    const rect = els.canvasWrapper.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Apply inverse transform: (screen_coord - translation) / scale
    return {
        x: (mouseX - state.view.x) / state.view.scale,
        y: (mouseY - state.view.y) / state.view.scale
    };
}

// --- Zoom & Pan ---

function handleWheel(e) {
    if (!state.imageObj) return;

    e.preventDefault(); // Prevent browser zooming/scrolling

    if (e.ctrlKey || e.metaKey) {
        // Zoom
        const rect = els.canvasWrapper.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Point under mouse in image coordinates before zoom
        const imgX = (mouseX - state.view.x) / state.view.scale;
        const imgY = (mouseY - state.view.y) / state.view.scale;

        const delta = -Math.sign(e.deltaY) * 0.1; // Zoom sensitivity
        let newScale = state.view.scale + delta;

        // Clamp zoom level
        if (newScale < 0.1) newScale = 0.1;
        if (newScale > 50.0) newScale = 50.0;

        // Calculate new view.x and view.y to keep imgX, imgY under mouseX, mouseY
        state.view.x = mouseX - imgX * newScale;
        state.view.y = mouseY - imgY * newScale;
        state.view.scale = newScale;

        updateTransform();
    } else {
        // Pan
        state.view.x -= e.deltaX;
        state.view.y -= e.deltaY;
        updateTransform();
    }
    redraw(); // Redraw annotations (e.g., line thickness)
}

function updateZoomStep(delta) {
    if (!state.imageObj) return;

    // Zoom centered on the current view's center
    const wrapperW = els.canvasWrapper.clientWidth;
    const wrapperH = els.canvasWrapper.clientHeight;

    const centerX = wrapperW / 2;
    const centerY = wrapperH / 2;

    // Point under center in image coordinates
    const imgX = (centerX - state.view.x) / state.view.scale;
    const imgY = (centerY - state.view.y) / state.view.scale;

    let newScale = state.view.scale + delta;
    if (newScale < 0.1) newScale = 0.1;
    if (newScale > 50.0) newScale = 50.0;

    state.view.x = centerX - imgX * newScale;
    state.view.y = centerY - imgY * newScale;
    state.view.scale = newScale;

    updateTransform();
    redraw();
}

function resetZoom() {
    fitImageToScreen();
}

// --- Interaction (Mouse) ---

const HANDLE_SIZE = 8; // Display size in screen pixels

function getHandleSizeInImageCoords() {
    return HANDLE_SIZE / state.view.scale;
}

// Check if point hits a handle of a box (in image coordinates)
function hitTestHandle(x, y, box) {
    const hs = getHandleSizeInImageCoords(); // handle size in image coords
    const half = hs / 2;

    // Handle positions (in image coordinates)
    const handles = {
        'resize-tl': { x: box.x, y: box.y },
        'resize-tr': { x: box.x + box.width, y: box.y },
        'resize-bl': { x: box.x, y: box.y + box.height },
        'resize-br': { x: box.x + box.width, y: box.y + box.height }
    };

    for (const [mode, pos] of Object.entries(handles)) {
        if (x >= pos.x - half && x <= pos.x + half &&
            y >= pos.y - half && y <= pos.y + half) {
            return mode;
        }
    }
    return null;
}

// Check if point is inside box (in image coordinates)
function hitTestBox(x, y, box) {
    return (x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height);
}

// --- Mode Switching ---

function setToolMode(mode) {
    state.toolMode = mode;

    // UI Update
    if (mode === 'draw') {
        els.modeDrawBtn.classList.add('active');
        els.modePanBtn.classList.remove('active');
        els.canvas.style.cursor = 'default';
        els.canvas.classList.remove('cursor-pan');
    } else {
        els.modePanBtn.classList.add('active');
        els.modeDrawBtn.classList.remove('active');
        els.canvas.style.cursor = 'grab';
        els.canvas.classList.add('cursor-pan');
        state.selectedBoxId = null; // Deselect when panning
        redraw();
    }
}

function handleMouseDown(e) {
    if (!state.imageObj) return;

    // PAN MODE Logic
    // Trigger if:
    // 1. toolMode is 'pan'
    // 2. Middle Mouse Button (button === 1)
    // 3. Spacebar is held (toolMode is temp 'pan')
    if (state.toolMode === 'pan' || e.button === 1) {
        e.preventDefault(); // prevent scroll/paste
        state.isPanning = true;
        state.panStartX = e.clientX;
        state.panStartY = e.clientY;
        state.viewStartX = state.view.x; // Store current view position
        state.viewStartY = state.view.y;
        if (e.button === 1) els.canvas.style.cursor = 'grabbing'; // visual override for mid-click
        else els.canvas.classList.add('cursor-panning');
        return;
    }

    // DRAW MODE (Left Click only)
    if (e.button !== 0) return;

    const pos = getMousePos(e); // Image coordinates
    const imageName = state.images[state.currentImageIndex];
    const boxes = state.annotations[imageName] || [];

    // 1. Check if we strictly hit a handle of the Selected Box first
    if (state.selectedBoxId) {
        const selBox = boxes.find(b => String(b.id) === String(state.selectedBoxId));
        if (selBox) {
            const handle = hitTestHandle(pos.x, pos.y, selBox);
            if (handle) {
                startDrag(handle, selBox, pos);
                return;
            }
        }
    }

    // 2. Check overlap logic
    // Reverse iteration to select "topmost" rendered box
    let hitBox = null;
    for (let i = boxes.length - 1; i >= 0; i--) {
        if (hitTestBox(pos.x, pos.y, boxes[i])) {
            hitBox = boxes[i];
            break;
        }
    }

    if (hitBox) {
        state.selectedBoxId = hitBox.id;
        state.currentLabel = hitBox.label; // Sync label
        els.labelInput.value = state.currentLabel;
        startDrag('move', hitBox, pos);
    } else {
        // 3. Start drawing new box
        state.selectedBoxId = null; // deselect
        state.isDrawing = true;
        state.dragMode = 'create';
        state.startX = pos.x; // Image coordinates
        state.startY = pos.y;
        redraw();
    }
}

function startDrag(mode, box, pos) {
    state.isDrawing = true;
    state.dragMode = mode;
    state.dragStartBox = { ...box }; // snapshot
    state.dragStartX = pos.x; // Image coordinates
    state.dragStartY = pos.y;
    redraw();
}

function handleMouseMove(e) {
    // PAN MODE
    if (state.isPanning) {
        e.preventDefault();
        const dx = e.clientX - state.panStartX;
        const dy = e.clientY - state.panStartY;
        state.view.x = state.viewStartX + dx;
        state.view.y = state.viewStartY + dy;
        updateTransform();
        return;
    }

    const pos = getMousePos(e); // Image coordinates

    // Update Cursor (Draw Mode)
    if (!state.isDrawing && state.toolMode === 'draw') {
        let cursor = 'default';
        const imageName = state.images[state.currentImageIndex];
        const boxes = state.annotations[imageName] || [];

        if (state.selectedBoxId) {
            const selBox = boxes.find(b => String(b.id) === String(state.selectedBoxId));
            if (selBox) {
                const handle = hitTestHandle(pos.x, pos.y, selBox);
                if (handle) {
                    if (handle === 'resize-tl' || handle === 'resize-br') cursor = 'nwse-resize';
                    else if (handle === 'resize-tr' || handle === 'resize-bl') cursor = 'nesw-resize';
                } else if (hitTestBox(pos.x, pos.y, selBox)) {
                    cursor = 'move';
                }
            }
        }

        if (cursor === 'default') {
            for (let i = boxes.length - 1; i >= 0; i--) {
                if (hitTestBox(pos.x, pos.y, boxes[i])) {
                    cursor = 'pointer';
                    break;
                }
            }
        }
        els.canvas.style.cursor = cursor;
        return; // Just hovering
    }

    // DRAWING ACTIONS
    if (!state.isDrawing) return; // Should not happen if logic is correct

    if (state.dragMode === 'create') {
        redraw(); // Redraw existing annotations
        const width = pos.x - state.startX;
        const height = pos.y - state.startY;

        ctx.beginPath();
        ctx.rect(state.startX, state.startY, width, height);
        ctx.lineWidth = 2 / state.view.scale; // Line width invariant to zoom
        ctx.strokeStyle = '#007acc';
        ctx.stroke();
        // ctx.fillStyle = 'rgba(0, 122, 204, 0.2)'; // No fill for draft
        // ctx.fill();
        return;
    }

    // Move/Resize Logic
    const imageName = state.images[state.currentImageIndex];
    if (!imageName) return;
    const box = state.annotations[imageName].find(b => String(b.id) === String(state.selectedBoxId));
    if (!box) return;

    // Movement Delta in image coordinates
    const dx = pos.x - state.dragStartX;
    const dy = pos.y - state.dragStartY;

    if (state.dragMode === 'move') {
        box.x = state.dragStartBox.x + dx;
        box.y = state.dragStartBox.y + dy;
    } else if (state.dragMode === 'resize-br') {
        box.width = state.dragStartBox.width + dx;
        box.height = state.dragStartBox.height + dy;
    } else if (state.dragMode === 'resize-tl') {
        box.x = state.dragStartBox.x + dx;
        box.y = state.dragStartBox.y + dy;
        box.width = state.dragStartBox.width - dx;
        box.height = state.dragStartBox.height - dy;
    } else if (state.dragMode === 'resize-tr') {
        box.y = state.dragStartBox.y + dy;
        box.width = state.dragStartBox.width + dx;
        box.height = state.dragStartBox.height - dy;
    } else if (state.dragMode === 'resize-bl') {
        box.x = state.dragStartBox.x + dx;
        box.width = state.dragStartBox.width - dx;
        box.height = state.dragStartBox.height + dy;
    }

    redraw();
}

function handleMouseUp(e) {
    // PAN MODE
    if (state.isPanning) {
        state.isPanning = false;
        els.canvas.classList.remove('cursor-panning');
        els.canvas.style.cursor = state.toolMode === 'pan' ? 'grab' : 'default';
        return;
    }

    if (!state.isDrawing) return;
    state.isDrawing = false;

    const imageName = state.images[state.currentImageIndex];

    if (state.dragMode === 'create') {
        const pos = getMousePos(e); // Image coordinates
        const width = pos.x - state.startX;
        const height = pos.y - state.startY;

        if (Math.abs(width) > 5 / state.view.scale && Math.abs(height) > 5 / state.view.scale) { // Minimum size in image coords
            // New Box
            let x = state.startX;
            let y = state.startY;
            let w = width;
            let h = height;

            if (w < 0) { x += w; w = Math.abs(w); }
            if (h < 0) { y += h; h = Math.abs(h); }

            const newBox = {
                id: Date.now().toString(), // String ID
                x: x,
                y: y,
                width: w,
                height: h,
                label: state.currentLabel
            };

            if (!state.annotations[imageName]) state.annotations[imageName] = [];
            state.annotations[imageName].push(newBox);
            state.selectedBoxId = newBox.id; // Select new box
        }
    } else {
        // Finished moving/resizing: normalize
        // Robust ID check
        const box = state.annotations[imageName].find(b => String(b.id) === String(state.selectedBoxId));
        if (box) {
            if (box.width < 0) { box.x += box.width; box.width = Math.abs(box.width); }
            if (box.height < 0) { box.y += box.height; box.height = Math.abs(box.height); }
        }
    }

    state.dragMode = null;
    state.dragStartBox = null;

    saveAnnotations(imageName);
    redraw();
}

// --- Rendering ---

function redraw() {
    if (!state.imageObj) return;

    // Clear canvas (which is now natural image size)
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);

    // Draw Image (1:1 on canvas, scaling handled by CSS transform)
    ctx.drawImage(state.imageObj, 0, 0, els.canvas.width, els.canvas.height);

    // Draw Annotations
    const imageName = state.images[state.currentImageIndex];
    if (imageName && state.annotations[imageName]) {
        state.annotations[imageName].forEach(box => {
            drawBox(box, String(box.id) === String(state.selectedBoxId));
        });
    }

    // Update delete button state
    els.deleteSelectedBtn.disabled = !state.selectedBoxId;
    if (state.selectedBoxId) els.deleteSelectedBtn.classList.remove('disabled-look');
    else els.deleteSelectedBtn.classList.add('disabled-look');
}

function drawBox(box, isSelected) {
    // Line width should be invariant of zoom, so scale it by inverse of view scale
    const lw = (isSelected ? 3 : 2) / state.view.scale;
    const hs = getHandleSizeInImageCoords(); // handle size in image coords
    const half = hs / 2;

    ctx.beginPath();
    ctx.rect(box.x, box.y, box.width, box.height);
    ctx.lineWidth = lw;
    ctx.strokeStyle = isSelected ? '#ffcc00' : '#5cb85c'; // Highlight selected
    ctx.stroke();

    // Fill
    ctx.fillStyle = isSelected ? 'rgba(255, 204, 0, 0.1)' : 'rgba(92, 184, 92, 0.2)';
    ctx.fill();

    // Handles if selected
    if (isSelected) {
        ctx.fillStyle = '#fff';
        // TL, TR, BL, BR
        ctx.fillRect(box.x - half, box.y - half, hs, hs);
        ctx.fillRect(box.x + box.width - half, box.y - half, hs, hs);
        ctx.fillRect(box.x - half, box.y + box.height - half, hs, hs);
        ctx.fillRect(box.x + box.width - half, box.y + box.height - half, hs, hs);

        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1 / state.view.scale; // Handle border also invariant
        ctx.strokeRect(box.x - half, box.y - half, hs, hs);
        ctx.strokeRect(box.x + box.width - half, box.y - half, hs, hs);
        ctx.strokeRect(box.x - half, box.y + box.height - half, hs, hs);
        ctx.strokeRect(box.x + box.width - half, box.y + box.height - half, hs, hs);
    }

    // Label
    const fontSize = 14 / state.view.scale; // Font size invariant to zoom
    ctx.font = `${fontSize}px Inter`;
    const label = box.label || state.currentLabel;
    const tw = ctx.measureText(label).width + (10 / state.view.scale); // Padding also invariant
    const textX = box.x;
    const textY = box.y - (20 / state.view.scale); // Position above box

    ctx.fillStyle = isSelected ? '#ffcc00' : '#5cb85c';
    ctx.fillRect(textX, textY, tw, 20 / state.view.scale);
    ctx.fillStyle = isSelected ? '#000' : '#fff';
    ctx.fillText(label, textX + (5 / state.view.scale), textY + (15 / state.view.scale));
}

function deleteSelectedBox() {
    console.log(`DELETE CALL: Sel=${state.selectedBoxId}`);
    logToDiv(`DELETE CALL: Sel=${state.selectedBoxId}`);

    if (!state.selectedBoxId) {
        logToDiv("Delete aborted: No selection");
        return;
    }

    const imageName = state.images[state.currentImageIndex];
    if (!state.annotations[imageName]) return;

    const initialLen = state.annotations[imageName].length;
    const targetId = String(state.selectedBoxId);

    console.log("Current Boxes:", state.annotations[imageName].map(b => `${b.id} (${typeof b.id})`));
    console.log("Target ID:", targetId, typeof targetId);

    // Filter
    const newBoxes = state.annotations[imageName].filter(b => String(b.id) !== targetId);
    state.annotations[imageName] = newBoxes;

    const finalLen = state.annotations[imageName].length;

    if (finalLen < initialLen) {
        logToDiv(`Successfully deleted box ${targetId}`);
        state.selectedBoxId = null;
        saveAnnotations(imageName);
        redraw();
    } else {
        logToDiv(`FAILED to delete. Target=${targetId}. IDs available: ${state.annotations[imageName].map(b => b.id).join(',')}`);
        console.error("Delete failed. ID mismatch?");
    }
}

function clearAnnotations() {
    if (!confirm("Clear all boxes for this image?")) return;
    const imageName = state.images[state.currentImageIndex];
    state.annotations[imageName] = [];
    state.selectedBoxId = null;
    saveAnnotations(imageName);
    redraw();
}

function deleteLastBox() {
    // Legacy support, or just call deleteSelected if exists, else pop
    if (state.selectedBoxId) {
        deleteSelectedBox();
    } else {
        const imageName = state.images[state.currentImageIndex];
        if (state.annotations[imageName] && state.annotations[imageName].length > 0) {
            state.annotations[imageName].pop();
            saveAnnotations(imageName);
            redraw();
        }
    }
}

async function handleAutoAnnotateAll() {
    const type = els.aaModelType.value;
    let prompt = null;
    let filename = null;

    if (type === 'countgd') {
        prompt = els.aaPrompt.value.trim();
        if (!prompt) {
            alert("Please enter a text prompt");
            return;
        }
    } else {
        filename = els.aaModelFile.value;
        if (!filename) {
            alert("Please select a model file");
            return;
        }
        prompt = "detected_object";
    }

    if (!confirm(`This will run auto-annotation with ${type.toUpperCase()} on ALL ${state.images.length} images. This may take a while. Continue?`)) return;

    els.aaBtnAll.disabled = true;
    els.aaBtn.disabled = true;

    const total = state.images.length;
    let processed = 0;
    const errors = [];

    els.aaStatus.innerText = `Starting batch process (0/${total})...`;

    // Base payload
    const basePayload = {
        confidence_thresh: parseFloat(els.aaConf.value),
        model_type: type,
        tiled: els.aaTiled.checked
    };
    if (prompt) basePayload.text_prompt = prompt;
    if (filename) basePayload.model_filename = filename;

    const customLabel = els.aaCustomLabel.value.trim();
    if (customLabel) basePayload.custom_label = customLabel;

    for (const imageName of state.images) {
        processed++;
        els.aaStatus.innerText = `Processing ${processed}/${total}: ${imageName}`;

        try {
            const payload = { ...basePayload, image_name: imageName };

            const res = await fetch('/api/auto_annotate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                const data = await res.json();
                const newBoxes = data.boxes;

                // Append
                if (!state.annotations[imageName]) state.annotations[imageName] = [];
                state.annotations[imageName].push(...newBoxes);
                await saveAnnotations(imageName);
            } else {
                errors.push(imageName);
                console.error(`Failed ${imageName}`);
            }
        } catch (err) {
            console.error(err);
            errors.push(imageName);
        }

        // Slight delay to allow UI update
        await new Promise(r => setTimeout(r, 50));
    }

    els.aaBtnAll.disabled = false;
    els.aaBtn.disabled = false;

    let msg = `Batch Complete. Processed ${total}.`;
    if (errors.length > 0) msg += ` Errors in ${errors.length} images.`;
    els.aaStatus.innerText = msg;

    // Refresh current view
    redraw();
}

function updateImageListUI() {
    els.imageList.innerHTML = '';
    els.imageCount.innerText = `(${state.images.length})`;

    state.images.forEach((name, i) => {
        const li = document.createElement('li');
        li.innerText = name;
        li.onclick = () => loadImage(i);
        els.imageList.appendChild(li);
    });
}

async function exportData() {
    if (state.images.length === 0) {
        alert("No images to export.");
        return;
    }

    // Show Modal
    els.exportModal.classList.remove('hidden');
    els.exportActions.classList.add('hidden'); // Ensure actions hidden on start
    updateExportProgress(0, "Starting export...");

    try {
        // Start Job
        const startRes = await fetch('/api/export/start', { method: 'POST' });
        if (!startRes.ok) throw new Error("Failed to start export");
        const { job_id } = await startRes.json();

        // Poll Status
        const pollInterval = setInterval(async () => {
            try {
                const statusRes = await fetch(`/api/export/status/${job_id}`);
                if (!statusRes.ok) return; // Skip this tick if network blip

                const job = await statusRes.json();

                // Update UI
                if (job.total > 0) {
                    const pct = Math.round((job.current / job.total) * 100);
                    updateExportProgress(pct, job.message);
                } else {
                    updateExportProgress(0, job.message);
                }

                if (job.status === 'completed') {
                    clearInterval(pollInterval);
                    // Trigger Download
                    window.location.href = `/api/export/download/${job_id}`;

                    // Update UI to reflect Auto-Cleanup
                    updateExportProgress(100, "Export Downloaded. Workspace Cleared.");

                    // Show actions? No, reset is done. Just refresh list after short delay.
                    setTimeout(async () => {
                        // Refresh image list to show empty state
                        await fetchImageList();
                        alert("Export downloaded & workspace cleared automatically!");
                        els.exportModal.classList.add('hidden');

                        // Also hide actions if they were visible
                        els.exportActions.classList.add('hidden');
                    }, 2000);

                } else if (job.status === 'failed') {
                    clearInterval(pollInterval);
                    alert("Export Failed: " + job.error);
                    els.exportModal.classList.add('hidden');
                }

            } catch (err) {
                console.error("Polling error", err);
            }
        }, 500);

    } catch (err) {
        console.error(err);
        alert("Export Error: " + err.message);
        els.exportModal.classList.add('hidden');
    }
}

async function handleResetDataset() {
    if (!confirm("Are you sure you want to RESET the dataset?\n\nThis will PERMANENTLY DELETE all current images and annotations.\n\nMake sure you have exported your data first!")) {
        return;
    }

    try {
        const res = await fetch('/api/reset_dataset', { method: 'POST' });
        if (!res.ok) throw new Error("Reset failed");

        await fetchImageList(); // Should return empty list

    } catch (err) {
        console.error("Reset error", err);
        alert("Failed to reset dataset: " + err.message);
    }
}

function updateExportProgress(pct, msg) {
    els.exportProgressBar.style.width = pct + "%";
    els.exportStatusText.innerText = `${msg} (${pct}%)`;
}

// Start
init();
