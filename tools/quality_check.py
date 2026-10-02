"""
Tool: check_image_quality
Purpose: decide whether an uploaded image is usable before running any
classification/segmentation on it.

Real checks (no model involved), in this order; the first one that fails
is reported so the user is told exactly what to fix:
  1. the file exists and can be opened as an image (not missing, corrupt
     or a non-image file)
  2. size            -- at least MIN_SIDE_PX on each side
  3. exposure        -- mean brightness within BRIGHTNESS_MIN..MAX
  4. contrast/detail -- not a flat or empty picture
  5. sharpness       -- variance of the Laplacian (measured at <= 512 px
                        wide, never upscaled, so small photos are not
                        penalised)
  6. skin visible    -- some skin-tone pixels (YCrCb range) are present

Thresholds were calibrated on the 18 sample photos in data/sample_images
(all pass; synthetic blurred / dark / over-exposed / flat images fail).
LIMITS: this cannot tell a burn from another skin condition (a facial
rash passes), and the skin-tone test is a coarse heuristic that was not
validated on darker skin tones, so its threshold is deliberately low.
"""

import os

import cv2
import numpy as np
from PIL import Image

MIN_SIDE_PX = 96
MAX_ANALYSIS_WIDTH = 512
BLUR_MIN = 20.0            # real samples >= 26.6, blurred copies <= 15.8
BRIGHTNESS_MIN = 45.0      # samples 97-195
BRIGHTNESS_MAX = 220.0
CONTRAST_MIN = 12.0        # samples >= 22.8
SKIN_FRACTION_MIN = 0.05   # samples >= 0.32
_SKIN_LOW = (0, 133, 77)
_SKIN_HIGH = (255, 173, 127)


def _result(verdict, issue, blur=None, brightness=None, contrast=None, skin=None):
    return {
        "blur_score": None if blur is None else round(float(blur), 1),
        "brightness": None if brightness is None else round(float(brightness), 1),
        "contrast": None if contrast is None else round(float(contrast), 1),
        "skin_fraction": None if skin is None else round(float(skin), 3),
        "verdict": verdict,
        "issue": issue,
    }


def check_image_quality(image_path: str) -> dict:
    """
    Args:
        image_path: path to the uploaded image file.

    Returns:
        dict with keys:
            blur_score, brightness, contrast, skin_fraction (float or None)
            verdict ("acceptable" | "retake_needed")
            issue (str or None) -- plain-language instruction when retake_needed
    """
    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        return _result("retake_needed", "I couldn't find that image file. Please upload the photo again.")

    try:
        with Image.open(image_path) as im:
            im.verify()
        # np.fromfile + imdecode also works for non-ASCII Windows paths, unlike cv2.imread
        bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        bgr = None
    if bgr is None:
        return _result("retake_needed", "That file could not be opened as a photo (unsupported or damaged file). "
                                        "Please upload a JPG or PNG image.")

    h, w = bgr.shape[:2]
    if min(h, w) < MIN_SIDE_PX:
        return _result("retake_needed", f"The photo is too small ({w}x{h} pixels). Please upload a larger photo "
                                        f"(at least {MIN_SIDE_PX} pixels on each side, ideally 500 or more).")

    scale = min(1.0, MAX_ANALYSIS_WIDTH / w)
    work = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else bgr
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    brightness, contrast = float(gray.mean()), float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    small = cv2.resize(bgr, (256, max(1, int(h * 256 / w))))
    skin = float(cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb), _SKIN_LOW, _SKIN_HIGH).mean() / 255.0)
    metrics = dict(blur=blur, brightness=brightness, contrast=contrast, skin=skin)

    if brightness < BRIGHTNESS_MIN:
        return _result("retake_needed", "The photo is too dark. Please retake it in bright, even light "
                                        "(daylight works well) without a filter.", **metrics)
    if brightness > BRIGHTNESS_MAX:
        return _result("retake_needed", "The photo is overexposed (washed out). Please retake it without "
                                        "direct glare or flash on the skin.", **metrics)
    if contrast < CONTRAST_MIN:
        return _result("retake_needed", "The photo looks flat or empty (almost no detail). Please retake it "
                                        "closer to the affected area.", **metrics)
    if blur < BLUR_MIN:
        return _result("retake_needed", "The photo looks blurry. Please hold the camera steady, tap to focus on "
                                        "the affected skin, and retake it.", **metrics)
    if skin < SKIN_FRACTION_MIN:
        return _result("retake_needed", "I can't see skin in this photo. Please retake it so the affected area of "
                                        "the body fills most of the frame.", **metrics)
    return _result("acceptable", None, **metrics)
