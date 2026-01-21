
import numpy as np
import tensorflow as tf
import os
from ml_model import DRModel

def debug_prediction():
    output_file = "inference_results.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Starting Inference Debug\n")
        
        def log(msg):
            print(msg)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

        # Initialize model
        log("initializing model...")
        dr_model = DRModel("converted_keras")
        
        # Create dummy inputs
        # 1. Random noise (normalized -1 to 1)
        random_img = np.random.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)
        
        # 2. Black image (0 normalized to -1)
        black_img = -1.0 * np.ones((1, 224, 224, 3)).astype(np.float32)
        
        # 3. White image (255 normalized to 1)
        white_img = 1.0 * np.ones((1, 224, 224, 3)).astype(np.float32)
        
        test_images = {
            "Random Noise": random_img,
            "Solid Black": black_img,
            "Solid White": white_img
        }
        
        CLASSES = [
            "No_DR",           # index 0
            "Mild",            # index 1
            "Moderate",        # index 2
            "Severe",          # index 3
            "Proliferate_DR"   # index 4
        ]
        
        for name, img in test_images.items():
            log(f"\n--- Testing with {name} ---")
            
            # Predict with each model manually to see raw outputs
            if not dr_model.model:
                log("No model loaded.")
                return

            # Single Model Prediction
            img_batch = np.expand_dims(img, axis=0) if img.ndim == 3 else img
            # Check input shape
            if img_batch.shape != (1, 224, 224, 3):
                 # Resize if needed or failed logic, but here we assume correct input
                 pass

            pred = dr_model.model.predict(img_batch, verbose=0)[0]
            pred_str = ", ".join([f"{p:.4f}" for p in pred])
            log(f"Raw Probabilities: [{pred_str}]")
            
            pred_index = np.argmax(pred)
            confidence = pred[pred_index]
            label = CLASSES[pred_index]
            
            log(f"Result: {label} ({confidence:.4f})")
            
    except Exception as e:
        import traceback
        with open("inference_results.txt", "a", encoding="utf-8") as f:
            f.write(f"An error occurred: {e}\n")
            f.write(traceback.format_exc())

if __name__ == "__main__":
    debug_prediction()
