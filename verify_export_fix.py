import os
import sys
import shutil
import json

# Ensure we can import main
sys.path.append(os.getcwd())

# Import the function to test
from main import cleanup_after_export, IMAGES_DIR, ANNOTATIONS_FILE

def test_cleanup():
    print(f"Testing cleanup logic...")
    print(f"IMAGES_DIR: {IMAGES_DIR}")
    
    # 1. Setup dummy data
    # Create dummy image
    dummy_img = os.path.join(IMAGES_DIR, "test_image.jpg")
    with open(dummy_img, "w") as f:
        f.write("dummy image content")
        
    # Create dummy annotation
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump({"test_image.jpg": []}, f)
        
    # Create dummy zip
    dummy_zip = os.path.join(os.getcwd(), "data", "test_export.zip")
    with open(dummy_zip, "w") as f:
        f.write("dummy zip content")
        
    print("Dummy data created.")
    
    # 2. Run cleanup
    print(f"Running cleanup_after_export({dummy_zip})...")
    cleanup_after_export(dummy_zip)
    
    # 3. Verify results
    # Zip should EXIST
    if os.path.exists(dummy_zip):
        print("SUCCESS: Zip file preserved.")
    else:
        print("FAILURE: Zip file was deleted!")
        
    # Image should be DELETED
    if not os.path.exists(dummy_img):
        print("SUCCESS: Image file deleted.")
    else:
        print("FAILURE: Image file still exists!")
        
    # Annotations should be EMPTY
    with open(ANNOTATIONS_FILE, 'r') as f:
        anns = json.load(f)
        if anns == {}:
            print("SUCCESS: Annotations reset.")
        else:
            print(f"FAILURE: Annotations not reset: {anns}")

    # Cleanup the dummy zip manually if it survived (as expected)
    if os.path.exists(dummy_zip):
        os.remove(dummy_zip)

if __name__ == "__main__":
    test_cleanup()
