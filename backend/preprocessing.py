import numpy as np
import cv2
from PIL import Image, ImageOps
import io

def preprocess_image(image_bytes: bytes):
    """
    Preprocess image for Teachable Machine model (Keras).
    Logic matches the user's provided snippet:
    1. Load image with PIL
    2. Convert to RGB
    3. ImageOps.fit (Resize & Center Crop) to 224x224
    4. Normalize ((img / 127.5) - 1)
    """
    # 1. Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. Resize & Crop (Teachable Machine standard)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # 3. Convert to Numpy Array
    image_array = np.asarray(image)

    # 4. Normalize
    # (image_array.astype(np.float32) / 127.5) - 1
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0

    # 5. Create Batch (1, 224, 224, 3)
    batch_img = np.expand_dims(normalized_image_array, axis=0)

    # Return:
    # - batch_img: for model
    # - image_array_bgr: for OpenCV visualization/saving (converted from RGB)
    # - is_noisy: Dummy flag (not checking noise anymore)
    
    # Convert RGB PIL array to BGR for OpenCV compatibility in other parts of the app
    image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    return batch_img, image_array_bgr, False
