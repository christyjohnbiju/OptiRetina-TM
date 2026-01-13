import cv2
import numpy as np
from PIL import Image
import io

def calculate_snr(img_array):
    """
    Calculate PSNR/SNR of the image.
    Using peak signal-to-noise ratio (PSNR) as a proxy for 'noise' check.
    If PSNR is low, image is noisy.
    """
    blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
    mse = np.mean((img_array - blurred) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def preprocess_image(image_bytes: bytes):
    """
    1. Check noise
    2. Gaussian filtering if noisy
    3. Resize to 224x224
    4. Normalize to [-1, 1] (CRITICAL)
    """
    # Load image as RGB
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)

    # 1. Noise Check
    psnr = calculate_snr(img_array)
    is_noisy = psnr < 30.0

    processed_img = img_array.copy()

    # 2. Conditional Gaussian Filtering
    if is_noisy:
        print(f"Image is noisy (PSNR: {psnr:.2f}dB). Applying Gaussian filter.")
        processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    else:
        print(f"Image is clean (PSNR: {psnr:.2f}dB). Skipping filtering.")

    # 3. Resize to 224x224
    processed_img_resized = cv2.resize(processed_img, (224, 224))

    # 4. Normalization for MobileNetV3: [-1, 1]
    # img_array / 127.5 - 1.0
    normalized_img = processed_img_resized.astype("float32")
    normalized_img = normalized_img / 127.5 - 1.0
    
    # Expand dims for batch: (1, 224, 224, 3)
    batch_img = np.expand_dims(normalized_img, axis=0)

    # Return batch (normalized) and display image (RGB, original size or resized? use resized for consistency)
    return batch_img, processed_img_resized, is_noisy
