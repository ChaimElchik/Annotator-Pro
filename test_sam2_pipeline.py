import sys
import torch
from detector_wrapper import DetectorWrapper

def test():
    dw = DetectorWrapper.get_instance()
    # Download SAM2 weights first manually for test
    from ultralytics.utils.downloads import attempt_download_asset
    attempt_download_asset("sam2_t.pt")
    
    # Needs CountGD weights in correct path, so must run in /Users/chaim/Desktop/AnnotatorV2/Github version/
    import urllib.request
    urllib.request.urlretrieve("https://ultralytics.com/images/zidane.jpg", "zidane.jpg")
    
    print("Running CountGD -> SAM2 (Grounded SAM2) inference via DetectorWrapper")
    try:
        res = dw.run_inference(
            image_path="zidane.jpg",
            model_type="sam2",
            model_path="sam2_t.pt",
            text_prompt="person",
            confidence=0.1
        )
        print("Result boxes:", len(res))
        if len(res) > 0:
            print("First box:", res[0])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test()
