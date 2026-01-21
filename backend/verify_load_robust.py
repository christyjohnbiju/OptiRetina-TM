
import tensorflow as tf
import os
import sys

# Import the actual class from ml_model
from ml_model import DRModel

def test_loading():
    print("Testing DRModel initialization...")
    try:
        dr_model = DRModel()
        
        print("\n--- Model Status ---")
        if dr_model.model:
            print("SUCCESS: Model loaded.")
            print(f"Input Shape: {dr_model.model.inputs[0].shape}")
        else:
            print("FAILURE: Model is None.")
            
        print("\n--- Classes Status ---")
        print(f"Loaded Classes: {dr_model.classes}")
        
        expected_classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
        
        # Check if classes match expected format (underscores)
        all_match = True
        for i, cls in enumerate(dr_model.classes):
            if " " in cls:
                print(f"WARNING: Class '{cls}' contains spaces. Expected underscores.")
                all_match = False
        
        if all_match:
            print("SUCCESS: All classes seem normalized (no spaces).")
            
    except Exception as e:
        print(f"FAILURE DURING INIT: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
