"""
Tool: classify_burn_severity
Real implementation -- EfficientNetV2-S classifier + reliability
engine (temperature-scaled confidence + entropy + 0.90 threshold gate).

NOTE ON TEMPERATURE SCALING: the saved .keras model outputs softmax
probabilities, not raw logits. True temperature scaling is defined on
logits. Since logits aren't separately exposed by this saved model, we
approximate them via log(probability), apply temperature there, then
re-softmax. This is a standard practical workaround but is NOT
mathematically identical to logit-level temperature scaling -- state
this explicitly as a caveat in the report, alongside the other
calibration caveats already documented.
"""

import os
import numpy as np
from tensorflow import keras

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "classifier",
    "EfficientNetV2S_BaselineStyle_91.42.keras"
)

CLASS_NAMES = ["1st_degree", "2nd_degree", "3rd_degree"]
TEMPERATURE = 1.1571
RELIABILITY_THRESHOLD = 0.90
INPUT_SIZE = (224, 224)

_model = None  # loaded lazily, once, on first use


def _load_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
    return _model


def _preprocess(image_path: str) -> np.ndarray:
    """
    Resize to 224x224 RGB, keep pixel values 0-255 (model has its own
    internal Rescaling layer -- do NOT divide by 255 here).
    """
    img = keras.utils.load_img(image_path, target_size=INPUT_SIZE, color_mode="rgb")
    arr = keras.utils.img_to_array(img)          # float32, values 0-255
    arr = np.expand_dims(arr, axis=0)             # add batch dimension
    return arr


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Approximate temperature scaling from softmax probabilities (see
    module docstring for the caveat on why this is an approximation).
    """
    eps = 1e-12
    approx_logits = np.log(probs + eps)
    scaled = approx_logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))   # numerically stable softmax
    return exp_scaled / np.sum(exp_scaled)


def _entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.sum(probs * np.log(probs + eps)))


def classify_burn_severity(image_path: str) -> dict:
    """
    Args:
        image_path: path to the quality-checked image.

    Returns:
        dict -- see module docstring / Doc 7 for full field meanings.
    """
    model = _load_model()
    img_array = _preprocess(image_path)

    raw_probs = model.predict(img_array, verbose=0)[0]   # shape (3,)
    predicted_index = int(np.argmax(raw_probs))
    raw_confidence = float(raw_probs[predicted_index])

    calibrated_probs = _apply_temperature(raw_probs, TEMPERATURE)
    calibrated_confidence = float(calibrated_probs[predicted_index])
    entropy = _entropy(calibrated_probs)

    reliability = "HIGH" if calibrated_confidence >= RELIABILITY_THRESHOLD else "LOW"
    decision = "PROCEED" if reliability == "HIGH" else "ABSTAIN"

    return {
        "predicted_severity": CLASS_NAMES[predicted_index],
        "predicted_class_index": predicted_index,
        "raw_confidence": raw_confidence,
        "calibrated_confidence": calibrated_confidence,
        "entropy": entropy,
        "reliability": reliability,
        "decision": decision,
        "reliability_threshold": RELIABILITY_THRESHOLD,
        "model": "EfficientNetV2-S",
        "calibration_temperature": TEMPERATURE,
        # Diagnostic only -- full per-class breakdown, not normally
        # needed by the agent, but useful for debugging low-confidence
        # cases like this one.
        "all_class_probabilities": {
            CLASS_NAMES[i]: float(calibrated_probs[i]) for i in range(len(CLASS_NAMES))
        },
    }