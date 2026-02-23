import cv2
import os
import glob

def extract_frames(video_path: str, output_dir: str, fps: float = 1.0, prefix: str = "frame") -> int:
    """
    Extracts frames from a video at a specified frame rate.
    
    Args:
        video_path: Path to the input video file.
        output_dir: Directory where extracted frames will be saved.
        fps: Frames per second to extract.
        prefix: Prefix for the extracted frame filenames.
        
    Returns:
        Number of frames extracted.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Clear existing frames in the output directory if it's strictly for this video import
    # For now, let's keep it additive or maybe we should clear it? 
    # The requirement says "Input should be a video or folder of images".
    # We'll assume the user wants to populate a workspace. 
    # Let's clear it to be safe and avoid mixing sessions for now, or maybe the API will handle that.
    # For this function, let's just extract.
    
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / fps)
    if frame_interval < 1:
        frame_interval = 1
        
    count = 0
    saved_count = 0
    success = True
    
    while success:
        success, image = vidcap.read()
        if success:
            if count % frame_interval == 0:
                # Save frame as JPEG
                frame_name = os.path.join(output_dir, f"{prefix}_{saved_count:05d}.jpg")
                cv2.imwrite(frame_name, image)
                saved_count += 1
            count += 1
            
    vidcap.release()
    return saved_count

def list_images(directory: str):
    """List all image files in a directory (case-insensitive)."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = []
    
    if not os.path.exists(directory):
        return []
        
    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            images.append(filename)
            
    # Sort for consistent order
    images.sort()
    return images
