"""
Face swap processor using InsightFace (inswapper_128) with optional
GFPGAN face enhancement.

Model weights are downloaded automatically on first use and cached
in the local `models/` directory.
"""

import os
import shutil
import cv2
import numpy as np
import requests
from pathlib import Path

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

INSWAPPER_PATH = MODELS_DIR / "inswapper_128.onnx"

# Public mirrors — tried in order until one succeeds
_INSWAPPER_URLS = [
    # Public HF mirror (no auth required)
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    # Fallback mirror
    "https://huggingface.co/theNeofr/inswapper/resolve/main/inswapper_128.onnx",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_inswapper() -> None:
    """Download inswapper_128.onnx.

    Strategy:
    1. huggingface_hub.hf_hub_download (uses HF_TOKEN env var automatically
       on HF Spaces — works if user has accepted gated-model terms).
    2. Plain HTTP fallback from public mirrors.
    """
    if INSWAPPER_PATH.exists() and INSWAPPER_PATH.stat().st_size > 100_000:
        return

    # ── Strategy 1: huggingface_hub ──────────────────────────────────────────
    try:
        from huggingface_hub import hf_hub_download
        print("[FaceSwapper] Downloading inswapper_128.onnx via HF Hub …")
        cached = hf_hub_download(
            repo_id="deepinsight/inswapper",
            filename="inswapper_128.onnx",
            token=os.environ.get("HF_TOKEN"),
        )
        shutil.copy(cached, INSWAPPER_PATH)
        print(f"[FaceSwapper] Saved to {INSWAPPER_PATH}")
        return
    except Exception as e:
        print(f"[FaceSwapper] HF Hub download failed ({e}), trying mirrors …")

    # ── Strategy 2: public mirrors ───────────────────────────────────────────
    for url in _INSWAPPER_URLS:
        try:
            print(f"[FaceSwapper] Trying {url} …")
            resp = requests.get(url, stream=True, timeout=180)
            resp.raise_for_status()
            with open(INSWAPPER_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            if INSWAPPER_PATH.stat().st_size > 500_000_000:  # ~554 MB expected
                print(f"[FaceSwapper] Saved to {INSWAPPER_PATH}")
                return
            INSWAPPER_PATH.unlink(missing_ok=True)
            print("[FaceSwapper] Mirror file too small, trying next …")
        except Exception as e:
            print(f"[FaceSwapper] Mirror failed ({e})")
            INSWAPPER_PATH.unlink(missing_ok=True)

    raise RuntimeError(
        "Could not download inswapper_128.onnx. "
        "Accept the model terms at https://huggingface.co/deepinsight/inswapper "
        "then add your HF token as a Space secret named HF_TOKEN."
    )


# ── Main class ────────────────────────────────────────────────────────────────

class FaceSwapper:
    """
    Swaps the dominant face from a source image onto every detected face in
    the target image.  Optionally runs GFPGAN to restore/enhance face quality.
    """

    def __init__(self):
        self._app = None        # InsightFace FaceAnalysis
        self._swapper = None    # inswapper ONNX model
        self._ready = False

    # ── Lazy initialisation ───────────────────────────────────────────────────

    def _init(self):
        if self._ready:
            return

        import insightface
        from insightface.app import FaceAnalysis

        # Face analysis (buffalo_l auto-downloads on first run)
        # Initialize face analysis (CPU-only for free HF Spaces tier)
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

        # inswapper model
        _download_inswapper()
        self._swapper = insightface.model_zoo.get_model(
            str(INSWAPPER_PATH),
            providers=["CPUExecutionProvider"],
        )

        self._ready = True

    # ── Enhancement (pure OpenCV, no extra models) ────────────────────────────

    @staticmethod
    def _enhance_opencv(image: np.ndarray, faces) -> np.ndarray:
        """
        For each detected face bounding box:
          1. Unsharp masking — recovers detail lost by inswapper's 128-px output
          2. CLAHE on the L channel — local contrast without blowing highlights
        """
        result = image.copy()
        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = (
                max(box[0], 0), max(box[1], 0),
                min(box[2], image.shape[1]), min(box[3], image.shape[0]),
            )
            if x2 <= x1 or y2 <= y1:
                continue

            roi = result[y1:y2, x1:x2].copy()

            # 1. Unsharp mask — scale radius with face size for consistent sharpness
            face_short = min(x2 - x1, y2 - y1)
            sigma = max(1.5, face_short / 80)  # larger face → larger radius
            blurred = cv2.GaussianBlur(roi, (0, 0), sigma)
            sharp = cv2.addWeighted(roi, 1.8, blurred, -0.8, 0)

            # 2. CLAHE on L channel
            lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Feather-blend back so edges stay smooth
            mask = np.zeros(roi.shape[:2], dtype=np.float32)
            pad = max(4, (y2 - y1) // 10)
            mask[pad:-pad, pad:-pad] = 1.0
            mask = cv2.GaussianBlur(mask, (0, 0), pad // 2 or 1)
            mask_3ch = mask[:, :, np.newaxis]
            result[y1:y2, x1:x2] = (
                enhanced_roi * mask_3ch + roi * (1 - mask_3ch)
            ).astype(np.uint8)

        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def swap(
        self,
        source_bgr: np.ndarray,
        target_bgr: np.ndarray,
        enhance: bool = True,
    ):
        """
        Swap the first detected face in *source_bgr* onto every face in
        *target_bgr*.

        Returns:
            (result_bgr, status_message)
        """
        self._init()

        try:
            # Preserve full resolution up to 2048px on the longest side.
            # Going beyond that adds little visible quality on CPU but multiplies time.
            MAX_DIM = 2048
            orig_h, orig_w = target_bgr.shape[:2]
            scale_down = 1.0
            if max(orig_h, orig_w) > MAX_DIM:
                scale_down = MAX_DIM / max(orig_h, orig_w)
                target_bgr = cv2.resize(
                    target_bgr,
                    (int(orig_w * scale_down), int(orig_h * scale_down)),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            source_faces = self._app.get(source_bgr)
            target_faces = self._app.get(target_bgr)

            if not source_faces:
                return None, "No face detected in source image."
            if not target_faces:
                return None, "No face detected in target image."

            source_face = source_faces[0]
            result = target_bgr.copy()

            for tgt_face in target_faces:
                result = self._swapper.get(
                    result, tgt_face, source_face, paste_back=True
                )

            # Always apply OpenCV enhancement — no extra deps needed
            if enhance:
                result = self._enhance_opencv(result, target_faces)

            # If we downscaled, upscale back to original resolution with Lanczos
            if scale_down < 1.0:
                result = cv2.resize(
                    result,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            return result, f"Swapped {len(target_faces)} face(s) successfully."

        except Exception as exc:
            return None, f"Face swap error: {exc}"

    def get_source_face(self, source_bgr: np.ndarray):
        """
        Detect and return the first face in *source_bgr*.
        Call once before a video loop and reuse the result in swap_frame().

        Returns:
            face object or None
        """
        self._init()
        faces = self._app.get(source_bgr)
        return faces[0] if faces else None

    def swap_frame(
        self,
        target_bgr: np.ndarray,
        source_face,
        cached_target_faces=None,
        enhance: bool = False,
    ):
        """
        Fast path for video — reuses a pre-computed source_face and optionally
        cached target faces (re-detection skipped when supplied).

        Returns:
            (result_bgr, target_faces_used)
        """
        self._init()

        # Cap video frames at 720p for speed; quality still good for motion
        MAX_VIDEO_DIM = 720
        orig_h, orig_w = target_bgr.shape[:2]
        scale_down = 1.0
        if max(orig_h, orig_w) > MAX_VIDEO_DIM:
            scale_down = MAX_VIDEO_DIM / max(orig_h, orig_w)
            target_bgr = cv2.resize(
                target_bgr,
                (int(orig_w * scale_down), int(orig_h * scale_down)),
                interpolation=cv2.INTER_LINEAR,
            )

        if cached_target_faces is None:
            target_faces = self._app.get(target_bgr)
        else:
            target_faces = cached_target_faces

        if not target_faces:
            return None, []

        result = target_bgr.copy()
        for tgt_face in target_faces:
            result = self._swapper.get(result, tgt_face, source_face, paste_back=True)

        if enhance:
            result = self._enhance_opencv(result, target_faces)

        # Scale back up to original frame size
        if scale_down < 1.0:
            result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return result, target_faces
