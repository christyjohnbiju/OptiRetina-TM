
import os
import tensorflow as tf
from ml_model import FixedDepthwiseConv2D

base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, "converted_keras", "keras_model.h5")

print(f"Attempting to load from: {path}")

try:
    model = tf.keras.models.load_model(
        path, 
        custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
        compile=False
    )
    print("SUCCESS: Model loaded.")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
