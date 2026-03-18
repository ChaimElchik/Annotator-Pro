import sys
import torch
from detector_wrapper import DetectorWrapper

def test():
    dw = DetectorWrapper.get_instance()
    
    print("\n--- Testing Standalone Grounding DINO ---")
    try:
        res_gd = dw.run_inference(
            image_path="zidane.jpg",
            model_type="groundingdino",
            text_prompt="person",
            confidence=0.1
        )
        print("GroundingDINO Result boxes:", len(res_gd))
        if len(res_gd) > 0:
            print("First box:", res_gd[0])
    except Exception as e:
        print("Error during Grounding DINO test:", e)

    print("\n--- Testing SAM2 (via Grounding DINO) ---")
    try:
        res_sam = dw.run_inference(
            image_path="zidane.jpg",
            model_type="sam2",
            model_path="sam2_t.pt",
            text_prompt="person",
            confidence=0.1
        )
        print("SAM2 Result boxes:", len(res_sam))
        if len(res_sam) > 0:
            print("First box:", res_sam[0])
    except Exception as e:
        print("Error during SAM2 test:", e)

if __name__ == "__main__":
    test()
