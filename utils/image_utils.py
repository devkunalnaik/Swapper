import cv2
import numpy as np
from PIL import Image


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR. Handles RGB, RGBA, palette, grayscale."""
    pil_img = pil_img.convert("RGB")  # always normalise to 3-channel RGB
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR numpy array to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))


def resize_to_max(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """Downscale image so its longest side does not exceed max_size."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def apply_color_correction(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Shift source pixel statistics (mean/std per LAB channel) inside the masked
    region to match those of the target, improving blending realism.
    """
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(float)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(float)

    mask_bool = mask > 128

    for ch in range(3):
        src_vals = source_lab[:, :, ch][mask_bool]
        tgt_vals = target_lab[:, :, ch][mask_bool]

        src_mean, src_std = src_vals.mean(), src_vals.std()
        tgt_mean, tgt_std = tgt_vals.mean(), tgt_vals.std()

        if src_std > 1e-6:
            source_lab[:, :, ch][mask_bool] = (
                (src_vals - src_mean) * (tgt_std / src_std) + tgt_mean
            )

    source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)


def feather_mask(mask: np.ndarray, blur_radius: int = 15) -> np.ndarray:
    """Apply Gaussian blur to soften mask edges."""
    if blur_radius % 2 == 0:
        blur_radius += 1
    return cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)


def alpha_blend(foreground: np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blend foreground onto background using a soft mask.

    Args:
        foreground: BGR image (same size as background).
        background: BGR image.
        mask: Single-channel uint8 mask (0-255).

    Returns:
        Blended BGR image.
    """
    alpha = mask.astype(float) / 255.0
    alpha_3ch = np.stack([alpha] * 3, axis=-1)
    blended = (foreground.astype(float) * alpha_3ch +
               background.astype(float) * (1.0 - alpha_3ch))
    return np.clip(blended, 0, 255).astype(np.uint8)
