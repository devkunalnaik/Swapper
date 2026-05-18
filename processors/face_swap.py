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

INSWAPPER_PATH  = MODELS_DIR / "inswapper_128.onnx"
CODEFORMER_PATH = MODELS_DIR / "codeformer.onnx"
ESPCN_PATH      = MODELS_DIR / "ESPCN_x2.pb"

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


def _download_codeformer() -> None:
    """Download CodeFormer ONNX model (~56 MB)."""
    if CODEFORMER_PATH.exists() and CODEFORMER_PATH.stat().st_size > 50_000_000:
        return
    urls = [
        "https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx",
    ]
    for url in urls:
        try:
            print(f"[FaceSwapper] Downloading CodeFormer from {url} …")
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            with open(CODEFORMER_PATH, "wb") as f:
                for chunk in resp.iter_content(65536):
                    f.write(chunk)
            if CODEFORMER_PATH.stat().st_size > 50_000_000:
                print("[FaceSwapper] CodeFormer ready.")
                return
            CODEFORMER_PATH.unlink(missing_ok=True)
        except Exception as e:
            print(f"[FaceSwapper] CodeFormer download failed: {e}")
            CODEFORMER_PATH.unlink(missing_ok=True)
    print("[FaceSwapper] CodeFormer unavailable — falling back to OpenCV enhancement.")


def _download_espcn() -> None:
    """Download ESPCN x2 super-resolution model (~100 KB)."""
    if ESPCN_PATH.exists() and ESPCN_PATH.stat().st_size > 50_000:
        return
    urls = [
        "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb",
    ]
    for url in urls:
        try:
            print(f"[FaceSwapper] Downloading ESPCN SR model from {url} …")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            ESPCN_PATH.write_bytes(resp.content)
            if ESPCN_PATH.stat().st_size > 50_000:
                print("[FaceSwapper] ESPCN SR model ready.")
                return
            ESPCN_PATH.unlink(missing_ok=True)
        except Exception as e:
            print(f"[FaceSwapper] ESPCN download failed: {e}")
            ESPCN_PATH.unlink(missing_ok=True)
    print("[FaceSwapper] ESPCN unavailable — skipping super-resolution step.")


# ── Main class ────────────────────────────────────────────────────────────────

class FaceSwapper:
    """
    Swaps the dominant face from a source image onto every detected face in
    the target image.  Optionally runs CodeFormer (ONNX) + ESPCN super-res
    for ultra-realistic high-definition output.
    """

    def __init__(self):
        self._app            = None   # InsightFace FaceAnalysis
        self._swapper        = None   # inswapper ONNX model
        self._codeformer     = None   # CodeFormer ONNX session
        self._sr             = None   # ESPCN DNN super-res (opencv-contrib)
        self._ready          = False

    # ── Lazy initialisation ───────────────────────────────────────────────────

    def _init(self):
        if self._ready:
            return

        import insightface
        from insightface.app import FaceAnalysis
        import onnxruntime as ort
        import multiprocessing

        n_threads = multiprocessing.cpu_count()

        # Use all available CPU cores for ONNX inference
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = n_threads
        sess_opts.inter_op_num_threads = n_threads
        sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        # Face analysis — 640 for images, 320 for video (set via swap_frame)
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

        # inswapper model with multi-thread session options
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

    # ── CodeFormer ONNX enhancement ───────────────────────────────────────────

    def _load_codeformer(self):
        """Lazy-load CodeFormer ONNX session. Returns None if unavailable."""
        if self._codeformer is not None:
            return self._codeformer
        try:
            _download_codeformer()
            if not CODEFORMER_PATH.exists():
                return None
            import onnxruntime as ort
            self._codeformer = ort.InferenceSession(
                str(CODEFORMER_PATH),
                providers=["CPUExecutionProvider"],
            )
            print("[FaceSwapper] CodeFormer ONNX loaded.")
        except Exception as e:
            print(f"[FaceSwapper] CodeFormer load failed: {e}")
            self._codeformer = None
        return self._codeformer

    def _load_sr(self):
        """Lazy-load ESPCN x2 DNN super-res (needs opencv-contrib). Returns None if unavailable."""
        if self._sr is not None:
            return self._sr
        try:
            _download_espcn()
            if not ESPCN_PATH.exists():
                return None
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(str(ESPCN_PATH))
            sr.setModel("espcn", 2)
            self._sr = sr
            print("[FaceSwapper] ESPCN 2× super-res loaded.")
        except Exception as e:
            print(f"[FaceSwapper] ESPCN load failed ({e}) — super-res disabled.")
            self._sr = None
        return self._sr

    def _enhance_codeformer(self, image: np.ndarray, faces) -> np.ndarray:
        """
        For each detected face:
          1. CodeFormer ONNX — neural face restoration at 512×512
          2. ESPCN 2× super-res — upscales small faces for HD output
          3. CLAHE — local contrast refinement
        Falls back to OpenCV enhancement if CodeFormer is unavailable.
        """
        sess = self._load_codeformer()
        if sess is None:
            return self._enhance_opencv(image, faces)

        sr   = self._load_sr()      # may be None — applied only when available
        result = image.copy()
        input_names = [i.name for i in sess.get_inputs()]

        for face in faces:
            box = face.bbox.astype(int)
            # Expand bbox 20% for realistic context padding
            bx1, by1, bx2, by2 = (
                max(box[0], 0), max(box[1], 0),
                min(box[2], image.shape[1]), min(box[3], image.shape[0]),
            )
            pad = int(min(bx2 - bx1, by2 - by1) * 0.15)
            x1 = max(0, bx1 - pad);  y1 = max(0, by1 - pad)
            x2 = min(image.shape[1], bx2 + pad); y2 = min(image.shape[0], by2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue

            roi  = result[y1:y2, x1:x2].copy()
            orig = roi.copy()
            h, w = roi.shape[:2]

            # ── 1. CodeFormer: BGR→RGB, resize to 512, normalize [-1, 1] ─────
            face_rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            face_512  = cv2.resize(face_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            inp       = (face_512.astype(np.float32) / 127.5) - 1.0   # [-1, 1]
            inp       = np.transpose(inp, (2, 0, 1))[np.newaxis]       # [1,3,512,512]

            try:
                out = sess.run(None, {input_names[0]: inp})[0]         # [1,3,512,512]
            except Exception as e:
                print(f"[FaceSwapper] CodeFormer inference failed: {e}")
                continue

            # Postprocess: [-1,1] → [0,255] → BGR
            out_rgb = np.squeeze(out)                                  # [3,512,512]
            out_rgb = np.transpose(out_rgb, (1, 2, 0))                 # [512,512,3]
            out_rgb = ((out_rgb + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

            # ── 2. ESPCN 2× super-res on small faces (<= 128 px) ─────────────
            if sr is not None and min(w, h) <= 128:
                try:
                    out_bgr = sr.upsample(out_bgr)
                    # Resize back to face region size (x2 upsample → scale back down)
                    out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
                except Exception:
                    out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
            else:
                out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

            # ── 3. CLAHE on L channel for final contrast refinement ───────────
            lab   = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # ── 4. Feather-blend onto result ──────────────────────────────────
            msk = np.zeros((h, w), dtype=np.float32)
            p   = max(4, min(h, w) // 10)
            msk[p:-p, p:-p] = 1.0
            msk = cv2.GaussianBlur(msk, (0, 0), p // 2 or 1)
            msk = msk[:, :, np.newaxis]
            result[y1:y2, x1:x2] = (
                out_bgr.astype(np.float32) * msk + orig.astype(np.float32) * (1 - msk)
            ).astype(np.uint8)

        return result

    # ── Laplacian pyramid blending ────────────────────────────────────────────

    @staticmethod
    def _face_ellipse_mask(shape: tuple, faces, expand: float = 0.35) -> np.ndarray:
        """
        Soft elliptical mask covering all detected face regions.
        255 = use swapped face, 0 = use original background.
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for face in faces:
            box = face.bbox.astype(int)
            x1 = max(box[0], 0);  y1 = max(box[1], 0)
            x2 = min(box[2], shape[1]); y2 = min(box[3], shape[0])
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ax = int(w // 2 * (1 + expand))
            ay = int(h // 2 * (1 + expand))
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        # Heavy Gaussian feather — wide transition = no visible seam
        blur = max(31, min(mask.shape[:2]) // 10)
        if blur % 2 == 0:
            blur += 1
        return cv2.GaussianBlur(mask, (blur, blur), 0)

    @staticmethod
    def _laplacian_blend(swapped: np.ndarray, original: np.ndarray,
                          mask: np.ndarray, levels: int = 6) -> np.ndarray:
        """
        Laplacian pyramid blending.
        Blends swapped face region onto original at multiple spatial scales
        so no hard edge is visible regardless of skin tone or lighting.

        mask: uint8 single-channel, 255 = take from swapped, 0 = take from original.
        """
        A = swapped.astype(np.float32)
        B = original.astype(np.float32)
        M = mask.astype(np.float32) / 255.0
        if M.ndim == 2:
            M = M[:, :, np.newaxis]
        # Expand to 3 channels so pyrDown/pyrUp never collapse the channel dim
        M = np.repeat(M, 3, axis=2)

        # Build Gaussian pyramids
        gA, gB, gM = [A], [B], [M]
        for _ in range(levels):
            gA.append(cv2.pyrDown(gA[-1]))
            gB.append(cv2.pyrDown(gB[-1]))
            gM.append(cv2.pyrDown(gM[-1]))

        # Build Laplacian pyramids
        lA, lB = [], []
        for i in range(levels):
            sz = (gA[i].shape[1], gA[i].shape[0])
            lA.append(gA[i] - cv2.pyrUp(gA[i + 1], dstsize=sz))
            lB.append(gB[i] - cv2.pyrUp(gB[i + 1], dstsize=sz))
        lA.append(gA[levels])
        lB.append(gB[levels])

        # Blend each level, reconstruct coarse→fine
        result = lA[levels] * gM[levels] + lB[levels] * (1.0 - gM[levels])
        for i in range(levels - 1, -1, -1):
            sz = (lA[i].shape[1], lA[i].shape[0])
            result = cv2.pyrUp(result, dstsize=sz) + lA[i] * gM[i] + lB[i] * (1.0 - gM[i])

        return np.clip(result, 0, 255).astype(np.uint8)

    # ── Public API ────────────────────────────────────────────────────────────

    def swap(
        self,
        source_bgr: np.ndarray,
        target_bgr: np.ndarray,
        enhance: bool = True,
        progress_cb=None,
    ):
        """
        Swap the first detected face in *source_bgr* onto every face in
        *target_bgr*.  Applies Laplacian pyramid blending for seamless edges.

        progress_cb: optional callable(fraction: float, label: str)

        Returns:
            (result_bgr, status_message)
        """
        def _p(v, msg):
            if progress_cb:
                progress_cb(v, msg)

        self._init()
        _p(0.1, "Models ready — detecting faces…")

        try:
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
            _p(0.3, "Source face detected — scanning target…")
            target_faces = self._app.get(target_bgr)

            if not source_faces:
                return None, "No face detected in source image."
            if not target_faces:
                return None, "No face detected in target image."

            _p(0.45, f"Swapping {len(target_faces)} face(s)…")
            source_face  = source_faces[0]
            result       = target_bgr.copy()
            original_bgr = target_bgr.copy()   # kept for Laplacian blend

            for tgt_face in target_faces:
                result = self._swapper.get(
                    result, tgt_face, source_face, paste_back=True
                )

            # ── Laplacian pyramid blending — removes hard boundary ─────────
            _p(0.65, "Blending edges (Laplacian pyramid)…")
            blend_mask = self._face_ellipse_mask(original_bgr.shape, target_faces)
            result     = self._laplacian_blend(result, original_bgr, blend_mask)

            # ── CodeFormer enhancement (images only) ──────────────────────
            if enhance:
                _p(0.80, "Enhancing quality (CodeFormer)…")
                result = self._enhance_codeformer(result, target_faces)

            # ── Upscale back to original resolution ───────────────────────
            if scale_down < 1.0:
                _p(0.95, "Upscaling to original resolution…")
                result = cv2.resize(
                    result,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            _p(1.0, f"Done — {len(target_faces)} face(s) swapped.")
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
            # Use smaller det_size for video to speed up detection
            self._app.det_model.input_size = (320, 320)
            target_faces = self._app.get(target_bgr)
            self._app.det_model.input_size = (640, 640)  # restore for images
        else:
            target_faces = cached_target_faces

        if not target_faces:
            return None, []

        result = target_bgr.copy()
        for tgt_face in target_faces:
            result = self._swapper.get(result, tgt_face, source_face, paste_back=True)

        # No per-frame enhancement for video — temporally unstable (causes flicker).
        # FFmpeg unsharp filter handles sharpening globally at encode time.

        # Scale back up to original frame size
        if scale_down < 1.0:
            result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return result, target_faces
