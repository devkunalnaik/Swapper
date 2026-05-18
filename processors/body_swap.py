"""
Body swap processor.

Pipeline
--------
1. Segment the person from both images with *rembg* (U²-Net).
2. Estimate body pose landmarks with *MediaPipe Pose*.
3. Compute an affine warp that maps the source torso keypoints onto the
   target torso keypoints, so the body roughly aligns with the target pose.
4. Blend the warped source body onto the target background using the
   segmentation mask + Gaussian feathering.  A Poisson seamless-clone pass
   is attempted for photorealistic colour blending.
"""

import cv2
import numpy as np
from PIL import Image

from utils.image_utils import (
    apply_color_correction,
    feather_mask,
    alpha_blend,
    resize_to_max,
)


# ── Landmark indices used for rough torso alignment ───────────────────────────
# MediaPipe Pose: 11=left shoulder, 12=right shoulder,
#                 23=left hip, 24=right hip
_TORSO_LANDMARKS = [11, 12, 23, 24]


class BodySwapper:
    """
    Replaces the body in *target_bgr* with the body from *source_bgr*.
    """

    def __init__(self):
        import mediapipe as mp

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _segment(self, bgr: np.ndarray) -> np.ndarray:
        """Return a uint8 single-channel person mask via rembg."""
        from rembg import remove

        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = remove(pil, only_mask=True)
        mask = np.array(result)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask

    def _get_landmarks(self, bgr: np.ndarray):
        """Return (x, y) pixel coords for all 33 pose landmarks, or None."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return None
        h, w = bgr.shape[:2]
        return [
            (int(lm.x * w), int(lm.y * h))
            for lm in results.pose_landmarks.landmark
        ]

    @staticmethod
    def _bbox(mask: np.ndarray):
        """Bounding box (x1, y1, x2, y2) of non-zero mask region, or None."""
        ys, xs = np.where(mask > 128)
        if len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _torso_centroid(landmarks: list) -> tuple[float, float] | None:
        """Return the mean (x, y) of the four torso landmarks."""
        pts = [landmarks[i] for i in _TORSO_LANDMARKS if i < len(landmarks)]
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return float(np.mean(xs)), float(np.mean(ys))

    # ── Public API ────────────────────────────────────────────────────────────

    def swap(
        self,
        source_bgr: np.ndarray,
        target_bgr: np.ndarray,
        blend_strength: float = 0.85,
    ) -> tuple[np.ndarray | None, str]:
        """
        Swap the source person's body into the target scene.

        Returns:
            (result_bgr, status_message)
        """
        try:
            # ── 1. Segment both images ────────────────────────────────────────
            src_mask = self._segment(source_bgr)
            tgt_mask = self._segment(target_bgr)

            src_bbox = self._bbox(src_mask)
            tgt_bbox = self._bbox(tgt_mask)

            if src_bbox is None:
                return None, "No person detected in source image."
            if tgt_bbox is None:
                return None, "No person detected in target image."

            sx1, sy1, sx2, sy2 = src_bbox
            tx1, ty1, tx2, ty2 = tgt_bbox

            # ── 2. Estimate poses ─────────────────────────────────────────────
            src_lm = self._get_landmarks(source_bgr)
            tgt_lm = self._get_landmarks(target_bgr)

            # ── 3. Scale source to match target bounding-box size ─────────────
            tgt_w, tgt_h = tx2 - tx1, ty2 - ty1

            src_person = source_bgr[sy1:sy2, sx1:sx2]
            src_mask_crop = src_mask[sy1:sy2, sx1:sx2]

            src_resized = cv2.resize(src_person, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(src_mask_crop, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)

            # ── 4. Optional pose-guided translation offset ────────────────────
            offset_x, offset_y = 0, 0
            if src_lm and tgt_lm:
                src_c = self._torso_centroid(src_lm)
                tgt_c = self._torso_centroid(tgt_lm)
                if src_c and tgt_c:
                    # Map source centroid into the cropped-then-resized space
                    scale_x = tgt_w / max(sx2 - sx1, 1)
                    scale_y = tgt_h / max(sy2 - sy1, 1)
                    src_cx_resized = (src_c[0] - sx1) * scale_x
                    src_cy_resized = (src_c[1] - sy1) * scale_y
                    # Offset in target coordinates
                    tgt_cx_in_roi = tgt_c[0] - tx1
                    tgt_cy_in_roi = tgt_c[1] - ty1
                    offset_x = int(tgt_cx_in_roi - src_cx_resized)
                    offset_y = int(tgt_cy_in_roi - src_cy_resized)

            # ── 5. Build a full-canvas composite ─────────────────────────────
            h_tgt, w_tgt = target_bgr.shape[:2]
            canvas_fg = np.zeros_like(target_bgr)
            canvas_mask = np.zeros((h_tgt, w_tgt), dtype=np.uint8)

            # Destination rectangle after applying offset
            dst_x1 = np.clip(tx1 + offset_x, 0, w_tgt)
            dst_y1 = np.clip(ty1 + offset_y, 0, h_tgt)
            dst_x2 = np.clip(tx1 + offset_x + tgt_w, 0, w_tgt)
            dst_y2 = np.clip(ty1 + offset_y + tgt_h, 0, h_tgt)

            # Corresponding source slice
            src_x1 = dst_x1 - (tx1 + offset_x)
            src_y1 = dst_y1 - (ty1 + offset_y)
            src_x2 = src_x1 + (dst_x2 - dst_x1)
            src_y2 = src_y1 + (dst_y2 - dst_y1)

            if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                return None, "Pose offset moved source body completely out of frame."

            canvas_fg[dst_y1:dst_y2, dst_x1:dst_x2] = src_resized[src_y1:src_y2, src_x1:src_x2]
            canvas_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask_resized[src_y1:src_y2, src_x1:src_x2]

            # ── 6. Color-correct source to match target's region ──────────────
            canvas_fg = apply_color_correction(canvas_fg, target_bgr, canvas_mask)

            # ── 7. Feather the mask and alpha-blend ───────────────────────────
            soft_mask = feather_mask(canvas_mask, blur_radius=21)
            soft_mask = (soft_mask.astype(float) * blend_strength).clip(0, 255).astype(np.uint8)

            result = alpha_blend(canvas_fg, target_bgr, soft_mask)

            # ── 8. Seamless clone for photorealistic blending (best-effort) ───
            try:
                center_x = int((dst_x1 + dst_x2) / 2)
                center_y = int((dst_y1 + dst_y2) / 2)
                # seamlessClone requires a clean binary mask
                sc_mask = (canvas_mask > 10).astype(np.uint8) * 255
                result = cv2.seamlessClone(
                    canvas_fg, target_bgr, sc_mask,
                    (center_x, center_y), cv2.NORMAL_CLONE,
                )
            except Exception as e:
                print(f"[BodySwapper] seamlessClone skipped: {e}")

            return result, "Body swap completed successfully."

        except Exception as exc:
            return None, f"Body swap error: {exc}"
