import tensorflow as tf
import numpy as np
import cv2
import os

# Fix for Teachable Machine Keras export compatibility
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        # Handle 'mask' or other arguments if passed
        return super().call(inputs)

class DRModel:
    def __init__(self, model_dir="converted_keras"):
        self.model = None
        self.model_dir = model_dir
        self.classes = [] # Dynamically loaded
        
        # Load the single Teachable Machine model
        self.load_model()
        self.load_labels()
        
        # For Grad-CAM (optional, validation needed if layer name differs)
        self.last_conv_layer_name = None
        if self.model:
            self.last_conv_layer_name = self.find_last_conv_layer(self.model)
            print("Last Conv Layer for Grad-CAM:", self.last_conv_layer_name)

    def load_labels(self):
        """
        Load labels from labels.txt in the model directory.
        Format: "0 ClassName"
        """
        print("DEBUG: Entering load_labels")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, self.model_dir, "labels.txt")
        
        if os.path.exists(path):
            try:
                temp_classes = {}
                with open(path, 'r') as f:
                    for line in f:
                        if line.strip():
                            # Split by first space: "0 No DR" -> ["0", "No DR"]
                            parts = line.strip().split(' ', 1)
                            if len(parts) == 2:
                                idx = int(parts[0])
                                # Normalize: "No DR" -> "No_DR" to match backend keys
                                label = parts[1].replace(" ", "_")
                                temp_classes[idx] = label
                
                # Sort by index to ensure correct order
                sorted_indices = sorted(temp_classes.keys())
                self.classes = [temp_classes[i] for i in sorted_indices]
                print(f"DEBUG: Loaded classes: {self.classes}")
            except Exception as e:
                print(f"DEBUG: Error loading labels: {e}")
                # Fallback if loading fails, though ideally we should fail hard or warn
                self.classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
        else:
            print(f"DEBUG: Labels file not found at {path}. Using default.")
            self.classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

    def load_model(self):
        print("DEBUG: Entering load_model")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, self.model_dir, "keras_model.h5")
        
        print(f"DEBUG: Loading model from: {path}")
        
        if os.path.exists(path):
            # Attempt 1: Standard load
            try:
                print("DEBUG: Attempting standard load_model...")
                # Map BOTH names just in case
                custom_objs = {
                    'DepthwiseConv2D': FixedDepthwiseConv2D,
                    'FixedDepthwiseConv2D': FixedDepthwiseConv2D
                }
                with tf.keras.utils.custom_object_scope(custom_objs):
                    self.model = tf.keras.models.load_model(path, compile=False)
                print(f"DEBUG: Model loaded successfully via load_model. Type: {type(self.model)}")
            except Exception as e:
                print(f"DEBUG: Standard load failed: {e}")
                print("DEBUG: Attempting manual build fallback...")
                try:
                    self.model = self._build_manual_model(path)
                    print("DEBUG: Manual build and weight load successful.")
                except Exception as e2:
                    print(f"DEBUG: Manual fallback failed: {e2}")
                    raise e # Raise original error or e2
        else:
            print(f"DEBUG: Error: Model file not found at {path}")
        print("DEBUG: Exiting load_model")

    def _build_manual_model(self, weights_path):
        """
        Manually reconstructs the TM architecture to bypass Keras 3 loading issues.
        Structure: Sequential([MobileNetV2_base, GAP, DenseHead])
        """
        # 1. Base (must match TM architecture: MobileNetV2 alpha=0.35)
        # We name it 'sequential_1' to match H5 weights
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            alpha=0.35, 
            include_top=False,
            weights=None,
            name='sequential_1'
        )
        
        # 2. GAP
        gap = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d_GlobalAveragePooling2D1')
        
        # 3. Head (sequential_3)
        # Note: Dense units must match TM export (defaults are 100, then NumClasses)
        head = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu', name='dense_Dense1'),
            tf.keras.layers.Dense(len(self.classes) if self.classes else 5, activation='softmax', name='dense_Dense2')
        ], name='sequential_3')
        
        # Assemble
        model = tf.keras.Sequential([
            base,
            gap,
            head
        ], name='sequential_4')
        
        # Load weights
        # by_name=True is critical. skip_mismatch=True handles minor version diffs.
        print("DEBUG: Loading weights into manual model...")
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        return model

    def find_last_conv_layer(self, model):
        """
        Attempt to find the last Conv2D layer for Grad-CAM.
        Teachable Machine usually exports a Sequential model with a Hub layer or just layers.
        We'll search recursively or linearly.
        """
        last_conv = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
            # If it's a nested model (like Functional inside Sequential), we might need to dig,
            # but usually TM Keras export is flat or standard.
        return last_conv

    def make_gradcam_heatmap(self, img_array, pred_index):
        if not self.model or not self.last_conv_layer_name:
             return None

        # Grad-CAM on the single model
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

    def predict(self, img_array, original_image_bgr):
        """
        Returns:
            label (str)
            confidence (float)
            gradcam_overlay (np.ndarray)
        """
        if not self.model:
            raise Exception("Model not loaded.")

        print(f"Running inference on shape: {img_array.shape}")
        
        # Inference
        prediction = self.model.predict(img_array)
        # prediction shape is (1, 5)
        
        print("\n--- INFERENCE RESULTS ---")
        print("Raw Probabilities:", prediction[0])
        
        pred_index = np.argmax(prediction[0])
        confidence = float(prediction[0][pred_index])
        
        # Use dynamically loaded classes
        if self.classes and pred_index < len(self.classes):
            label = self.classes[pred_index]
        else:
            label = "Unknown"
            
        print(f"Predicted: {label} ({confidence:.4f})")
        print("-------------------------\n")

        # Grad-CAM
        heatmap = None
        try:
            heatmap = self.make_gradcam_heatmap(img_array, pred_index)
            if heatmap is not None:
                heatmap = cv2.resize(
                    heatmap,
                    (original_image_bgr.shape[1], original_image_bgr.shape[0])
                )
        except Exception as e:
            print(f"Grad-CAM warning: {e}")
            heatmap = None

        if heatmap is not None:
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # heatmap is BGR from applyColorMap, original_image_bgr is BGR
            
            overlay = cv2.addWeighted(
                original_image_bgr, 0.6,
                heatmap, 0.4,
                0
            )
        else:
             overlay = original_image_bgr

        return label, confidence, overlay
