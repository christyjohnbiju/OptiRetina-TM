
import os
import cv2
import numpy as np
from ml_model import DRModel
from preprocessing import preprocess_image

def create_dummy_image():
    # Create a random image 224x224x3
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

def test_inference():
    print("Initializing Model...")
    dr_model = DRModel()
    
    print("Creating dummy image...")
    img_bytes = create_dummy_image()
    
    print("Preprocessing...")
    batch_img, processed_img_cv2, _ = preprocess_image(img_bytes)
    print(f"Batch Shape: {batch_img.shape}")
    
    print("Predicting...")
    label, confidence, overlay = dr_model.predict(batch_img, processed_img_cv2)
    
    print(f"Result: {label} ({confidence})")
    print("SUCCESS: Inference complete.")

if __name__ == "__main__":
    test_inference()
