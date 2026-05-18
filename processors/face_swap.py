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
            if INSWAPPER_PATH.stat().st_size > 100_000:
                print(f"[FaceSwapper] Saved to {INSWAPPER_PATH}")
                return
            INSWAPPER_PATH.unlink(missing_ok=True)
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
        self._enhancer = None   # GFPGAN (lazy)
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

    def _get_enhancer(self):
        """Lazy-load GFPGAN enhancer."""
        if self._enhancer is None:
            from gfpgan import GFPGANer

            self._enhancer = GFPGANer(
                model_path=(
                    "https://github.com/TencentARC/GFPGAN/releases/download/"
                    "v1.3.0/GFPGANv1.4.pth"
                ),
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
        return self._enhancer

    # ── Public API ────────────────────────────────────────────────────────────

    def swap(
        self,
        source_bgr: np.ndarray,
        target_bgr: np.ndarray,
        enhance: bool = False,
    ) -> tuple[np.ndarray | None, str]:
        """
        Swap the first detected face in *source_bgr* onto every face in
        *target_bgr*.

        Returns:
            (result_bgr, status_message)
        """
        self._init()

        try:
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

            if enhance:
                try:
                    _, _, result = self._get_enhancer().enhance(
                        result,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                    )
                except Exception as e:
                    print(f"[FaceSwapper] Enhancement skipped: {e}")

            return result, f"Swapped {len(target_faces)} face(s) successfully."

        except Exception as exc:
            return None, f"Face swap error: {exc}"
