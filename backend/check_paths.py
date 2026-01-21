
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = "converted_keras"
path = os.path.join(base_dir, model_dir, "keras_model.h5")

print(f"Base Dir: {base_dir}")
print(f"Constructed Path: {path}")
print(f"Exists? {os.path.exists(path)}")

# List contents of converted_keras
try:
    print(f"Contents of {model_dir}: {os.listdir(os.path.join(base_dir, model_dir))}")
except Exception as e:
    print(f"Error listing dir: {e}")
