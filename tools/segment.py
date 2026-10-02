"""
Tool: segment_wound_area
Real implementation -- SegFormer-B1 (nvidia/mit-b1, fine-tuned,
2-class: background/wound). No cm^2 output -- per project decision,
no scale-reference dataset exists, so this tool reports RELATIVE
spatial characteristics only (wound_fraction, connected_components),
used by the agent for triage evidence and by future reliability
analysis -- NOT a real-world area measurement.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation

from dotenv import load_dotenv

load_dotenv()   # so HF_TOKEN is available even when this module is
                # imported standalone (e.g. via test.py), not only
                # when agent/loop.py has already called load_dotenv()

# huggingface_hub reads this exact environment variable name
# automatically -- setting it here is enough, no extra code needed
# in the from_pretrained() call below.
if os.environ.get("HF_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "segformer", "segformer_b1_best.pth"
)
BASE_MODEL_NAME = "nvidia/mit-b1"
INPUT_SIZE = (384, 384)

# Standard ImageNet normalization, as confirmed used in training
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_preprocess_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Descriptive bands the agent uses instead of inventing its own thresholds.
# wound_fraction is relative to the PHOTO (not body surface), so these are
# coarse triage aids, not measurements. Change them here, in one place.
AREA_SMALL_MAX = 0.15          # below this: "small"
AREA_LARGE_MIN = 0.40          # at or above this: "large"; in between: "moderate"
SIGNIFICANT_REGION_FRACTION = 0.02   # a region counts as substantial if >= 2% of the image

_model = None  # loaded lazily, once, on first use


def _load_model():
    global _model
    if _model is None:
        model = SegformerForSemanticSegmentation.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

        # The .pth file is a full training checkpoint (epoch, metrics,
        # etc.), not a raw state_dict -- the actual weights are nested
        # under "model_state_dict".
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint  # fallback, in case it IS a raw state_dict

        model.load_state_dict(state_dict)
        model.eval()
        _model = model
    return _model


def _label_regions(mask: np.ndarray):
    """
    Label connected wound regions with scipy.ndimage.label (4-connectivity).
    Returns (number_of_regions, list_of_region_sizes_in_pixels).
    """
    from scipy import ndimage
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return 0, []
    sizes = ndimage.sum(mask, labeled, index=range(1, num_features + 1))
    return int(num_features), [int(x) for x in sizes]


def _area_band(wound_fraction: float, wound_pixels: int) -> str:
    if wound_pixels == 0:
        return "none"
    if wound_fraction < AREA_SMALL_MAX:
        return "small"
    if wound_fraction >= AREA_LARGE_MIN:
        return "large"
    return "moderate"


def segment_wound_area(image_path: str) -> dict:
    """
    Args:
        image_path: path to the quality-checked image.

    Returns:
        dict with keys:
            wound_pixels (int)
            image_pixels (int)
            wound_fraction (float)
            connected_components (int)      -- raw count, includes tiny specks
            area_band (str)                 -- "none" | "small" | "moderate" | "large"
            significant_regions (int)       -- regions >= 2% of the image
            multi_region (bool)             -- more than one significant region
            mask_available (bool)
    """
    model = _load_model()

    img = Image.open(image_path).convert("RGB")
    input_tensor = _preprocess_transform(img).unsqueeze(0)   # (1, 3, 384, 384)

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        logits = outputs.logits   # shape (1, 2, H/4, W/4) roughly

        # Bilinear-upsample logits to 384x384 BEFORE argmax, per the
        # confirmed evaluation pipeline (this matches the validated
        # Dice=0.8479 / IoU=0.7530 numbers).
        upsampled = F.interpolate(
            logits, size=INPUT_SIZE, mode="bilinear", align_corners=False
        )
        predicted = torch.argmax(upsampled, dim=1)   # shape (1, 384, 384)

    mask = predicted.squeeze(0).cpu().numpy().astype(np.uint8)  # (384, 384), values {0,1}

    wound_pixels = int(np.sum(mask == 1))
    image_pixels = int(mask.size)
    wound_fraction = wound_pixels / image_pixels if image_pixels > 0 else 0.0
    num_components, region_sizes = _label_regions(mask)
    significant = sum(1 for size in region_sizes if size >= SIGNIFICANT_REGION_FRACTION * image_pixels)

    return {
        "wound_pixels": wound_pixels,
        "image_pixels": image_pixels,
        "wound_fraction": wound_fraction,
        "connected_components": num_components,
        "area_band": _area_band(wound_fraction, wound_pixels),
        "significant_regions": significant,
        "multi_region": significant >= 2,
        "mask_available": True,
    }