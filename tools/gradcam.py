"""
Tool: generate_gradcam
Real implementation -- Grad-CAM on EfficientNetV2-S's final conv
feature map (top_conv, 7x7x1280), overlaid with
cv2.applyColorMap(COLORMAP_JET), saved under outputs/gradcam/.
Explainability only -- NOT part of the reliability score (see Doc 7:
CAM/segmentation overlap was tested and found non-discriminative,
AUC~0.53, so it's excluded from scoring).

IMPLEMENTATION NOTE: the classifier wraps EfficientNetV2-S as a NESTED
sub-model layer (visible via model.summary()). Keras 3 does not allow
splicing a tensor from INSIDE a nested sub-model into a new spliced
Functional Model (this fails with "Output ... is not connected to
inputs"). Instead, we manually chain layer calls inside the
GradientTape: call the backbone once to get its feature-map output
(which, since it feeds directly into global_average_pooling2d in the
outer model, IS the desired conv feature map -- functionally
equivalent to "top_conv" for CAM purposes), then manually replay the
remaining head layers (GAP -> BN -> Dense -> ... ) on top of it,
all as plain tensor operations the tape can track.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

from agent.userctx import current_user_id, user_dir
from tools.classify import _load_model, _preprocess, CLASS_NAMES, INPUT_SIZE

BACKBONE_LAYER_NAME = "efficientnetv2-s"
TARGET_LAYER = "top_conv"   # reported name; see note above on how it's actually obtained
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "gradcam")

# Names of the layers AFTER the backbone, in exact order, as seen in
# your model.summary() output:
# ['input_layer_1', 'efficientnetv2-s', 'global_average_pooling2d',
#  'batch_normalization', 'dense', 'batch_normalization_1', 'dropout',
#  'dense_1', 'dropout_1', 'dense_2']
_HEAD_LAYER_NAMES = [
    "global_average_pooling2d",
    "batch_normalization",
    "dense",
    "batch_normalization_1",
    "dropout",
    "dense_1",
    "dropout_1",
    "dense_2",
]


def generate_gradcam(image_path: str, predicted_class: str) -> dict:
    """
    Args:
        image_path: path to the original image.
        predicted_class: class name returned by classify_burn_severity
                          (must match one of CLASS_NAMES).

    Returns:
        dict with keys:
            method (str)
            target_layer (str)
            heatmap_path (str)
            overlay_path (str)
    """
    out_dir = user_dir("gradcam") if current_user_id() else OUTPUT_DIR   # per-user folder when signed in
    os.makedirs(out_dir, exist_ok=True)

    class_index = CLASS_NAMES.index(predicted_class)
    base_model = _load_model()
    backbone = base_model.get_layer(BACKBONE_LAYER_NAME)
    img_array = _preprocess(image_path)
    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        conv_output = backbone(img_tensor, training=False)
        tape.watch(conv_output)

        # Manually replay the remaining head layers, in order, on top
        # of the backbone's feature map -- this keeps everything
        # inside one tape so gradients flow correctly.
        x = conv_output
        for layer_name in _HEAD_LAYER_NAMES:
            layer = base_model.get_layer(layer_name)
            x = layer(x, training=False)
        predictions = x

        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_output)               # d(class_score)/d(conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))           # importance weight per channel

    conv_output_np = conv_output[0]                                 # drop batch dim -> (7, 7, 1280)
    heatmap = tf.reduce_sum(conv_output_np * pooled_grads, axis=-1) # weighted sum -> (7, 7)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-12)  # ReLU + normalize
    heatmap = heatmap.numpy()

    # Resize heatmap up to the original input size and colorize
    heatmap_resized = cv2.resize(heatmap, INPUT_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Load original image (as seen by the model, 224x224) to overlay on
    original = cv2.imread(image_path)
    original_resized = cv2.resize(original, INPUT_SIZE)
    overlay = cv2.addWeighted(original_resized, 0.6, colored_heatmap, 0.4, 0)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    heatmap_path = os.path.join(out_dir, f"{base_name}_heatmap.png")
    overlay_path = os.path.join(out_dir, f"{base_name}_overlay.png")

    cv2.imwrite(heatmap_path, colored_heatmap)
    cv2.imwrite(overlay_path, overlay)

    return {
        "method": "Grad-CAM",
        "target_layer": TARGET_LAYER,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
    }