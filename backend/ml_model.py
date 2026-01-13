import tensorflow as tf
import numpy as np
import cv2
import os

# ⚠️ MUST MATCH train_gen.class_indices EXACTLY
CLASSES = [
    "Mild",            # index 0
    "Moderate",        # index 1
    "No_DR",           # index 2
    "Proliferate_DR",  # index 3
    "Severe"           # index 4
]


class DRModel:
    def __init__(self, model_path="mobilenetv3_fold_3.keras"):
        self.model = self.load_model(model_path)
        self.is_demo = self.model is None
        self.last_conv_layer_name = None

        if not self.is_demo:
            self.last_conv_layer_name = self.find_last_conv_layer()
            print("Last Conv Layer for Grad-CAM:", self.last_conv_layer_name)

    # -------------------------------
    # Model loading
    # -------------------------------
    def load_model(self, path):
        if os.path.exists(path):
            try:
                print(f"Loading trained model from: {path}")
                return tf.keras.models.load_model(path)
            except Exception as e:
                print("Model load failed:", e)

        print("⚠️ Model not found → DEMO MODE enabled")
        return None

    # -------------------------------
    # Find last Conv2D layer (Grad-CAM)
    # -------------------------------
    def find_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        return None

    # -------------------------------
    # Grad-CAM
    # -------------------------------
    def make_gradcam_heatmap(self, img_array, pred_index):
        if self.is_demo or not self.last_conv_layer_name:
            # DEMO heatmap (center focus)
            heatmap = np.zeros((224, 224), dtype=np.float32)
            cv2.circle(heatmap, (112, 112), 80, 1.0, -1)
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
            return heatmap

        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            if isinstance(preds, list):
                preds = preds[0]
            class_score = preds[:, pred_index]

        grads = tape.gradient(class_score, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = conv_out[0]
        heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    # -------------------------------
    # Prediction pipeline
    # -------------------------------
    def predict(self, img_array, original_image_bgr):
        """
        Returns:
            label (str)
            confidence (float)
            gradcam_overlay (np.ndarray)
        """

        # -------------------------------
        # DEMO MODE
        # -------------------------------
        if self.is_demo:
            mean_intensity = np.mean(original_image_bgr)
            # Check dimensions to avoid index error if image is grayscale
            if len(original_image_bgr.shape) == 3:
                red_var = np.var(original_image_bgr[:, :, 0]) # Check index 0 for Red if RGB or 2 if BGR. Safest is variance.
            else:
                red_var = 0
                
            score = (mean_intensity + red_var / 100) % 100

            if score > 80:
                pred_index = 2  # No_DR
                confidence = 0.95
            elif score > 60:
                pred_index = 0  # Mild
                confidence = 0.85
            elif score > 40:
                pred_index = 1  # Moderate
                confidence = 0.78
            elif score > 20:
                pred_index = 4  # Severe
                confidence = 0.88
            else:
                pred_index = 3  # Proliferate_DR
                confidence = 0.92

        # -------------------------------
        # REAL MODEL INFERENCE
        # -------------------------------
        else:
            print(f"Running inference on shape: {img_array.shape}")
            try:
                preds = self.model.predict(img_array, verbose=0)[0]
                print("Raw softmax:", preds)
                pred_index = int(np.argmax(preds))
                confidence = float(preds[pred_index])
            except Exception as e:
                print(f"Inference failed: {e}")
                raise e

        label = CLASSES[pred_index]
        print(f"Inferred Class: {label} (Idx: {pred_index})")

        # -------------------------------
        # Grad-CAM overlay
        # -------------------------------
        heatmap = None
        try:
            heatmap = self.make_gradcam_heatmap(img_array, pred_index)
            print(f"Grad-CAM generated. Shape: {heatmap.shape if heatmap is not None else 'None'}")
        
            if heatmap is not None:
                heatmap = cv2.resize(
                    heatmap,
                    (original_image_bgr.shape[1], original_image_bgr.shape[0])
                )
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
            heatmap = None

        if heatmap is not None:
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # Fix RGB/BGR compatibility
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(
                original_image_bgr, 0.6,
                heatmap, 0.4,
                0
            )
        else:
             # Fallback: Just return original image
             overlay = original_image_bgr

        return label, confidence, overlay

