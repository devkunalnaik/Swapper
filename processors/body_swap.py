"""
Body swap processor.

Pipeline
--------
1. Segment both images with *rembg* (U²-Net) to isolate person masks.
2. Compute bounding boxes from the masks.
3. Scale the source person to match the target bounding-box dimensions.
4. Color-correct the source region to match target lighting/tone.
5. Feather-blend using the segmentation mask.
6. Apply Poisson seamless-clone for photorealistic edge merging.

MediaPipe is intentionally NOT used — its API changed in 0.10.14
(removed `solutions`) which breaks on Python 3.13.  Bounding-box
alignment alone is sufficient for clean body swaps.
"""

import cv2
import numpy as np
from PIL import Image

from utils.image_utils import (
    apply_color_correction,
    feather_mask,
    alpha_blend,
)


class BodySwapper:
    """
    Replaces the body in *target_bgr* with the body from *source_bgr*.
    """

    _rembg_session = None   # shared across instances; loaded once

    @classmethod
    def _get_session(cls):
        """Lazy-load the u2net_human_seg rembg session (better than general u2net)."""
        if cls._rembg_session is None:
            from rembg import new_session
            print("[BodySwapper] Loading u2net_human_seg segmentation model …")
            cls._rembg_session = new_session("u2net_human_seg")
        return cls._rembg_session

    # ── Private helpers ─────────────────────────────────────────────────

    @classmethod
    def _segment(cls, bgr: np.ndarray) -> np.ndarray:
        """Return uint8 single-channel person mask via rembg u2net_human_seg."""
        from rembg import remove

        pil    = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = remove(pil, only_mask=True, session=cls._get_session())
        mask   = np.array(result)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask

    @staticmethod
    def _bbox(mask: np.ndarray):
        """Bounding box (x1, y1, x2, y2) of the non-zero region, or None."""
        ys, xs = np.where(mask > 128)
        if len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _vertical_center_of_mass(mask: np.ndarray) -> float:
        """Y coordinate of the mask centre of mass (for vertical alignment)."""
        ys, _ = np.where(mask > 128)
        return float(ys.mean()) if len(ys) > 0 else mask.shape[0] / 2.0

    # ── Public API ────────────────────────────────────────────────────────────

    def swap(self, source_bgr, target_bgr, blend_strength=0.85):
        """
        Swap the source person's body into the target scene.

        Returns:
            (result_bgr, status_message)
        """
        try:
            # ── 1. Segment ────────────────────────────────────────────────────
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
            tgt_w, tgt_h = tx2 - tx1, ty2 - ty1

            # ── 2. Crop + resize source to target dimensions ──────────────────
            src_person   = source_bgr[sy1:sy2, sx1:sx2]
            src_mask_roi = src_mask[sy1:sy2, sx1:sx2]

            src_resized  = cv2.resize(src_person,   (tgt_w, tgt_h), interpolation=cv2.INTER_LANCZOS4)
            mask_resized = cv2.resize(src_mask_roi, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)

            # ── 3. Vertical CoM alignment ─────────────────────────────────────
            src_com_y = self._vertical_center_of_mass(src_mask_roi)
            tgt_com_y = self._vertical_center_of_mass(tgt_mask[ty1:ty2, tx1:tx2])
            scale_y   = tgt_h / max(sy2 - sy1, 1)
            offset_y  = int(tgt_com_y - src_com_y * scale_y)

            # ── 4. Composite onto full canvas ─────────────────────────────────
            h_t, w_t  = target_bgr.shape[:2]
            canvas_fg   = np.zeros_like(target_bgr)
            canvas_mask = np.zeros((h_t, w_t), dtype=np.uint8)

            dst_x1 = int(np.clip(tx1,          0, w_t))
            dst_y1 = int(np.clip(ty1 + offset_y, 0, h_t))
            dst_x2 = int(np.clip(tx1 + tgt_w,  0, w_t))
            dst_y2 = int(np.clip(ty1 + offset_y + tgt_h, 0, h_t))

            src_x1 = dst_x1 - tx1
            src_y1 = dst_y1 - (ty1 + offset_y)
            src_x2 = src_x1 + (dst_x2 - dst_x1)
            src_y2 = src_y1 + (dst_y2 - dst_y1)

            if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                return None, "Alignment offset moved body out of frame."

            canvas_fg  [dst_y1:dst_y2, dst_x1:dst_x2] = src_resized [src_y1:src_y2, src_x1:src_x2]
            canvas_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask_resized[src_y1:src_y2, src_x1:src_x2]

            # ── 5. Color correction ───────────────────────────────────────────
            canvas_fg = apply_color_correction(canvas_fg, target_bgr, canvas_mask)

            # ── 6. Feathered alpha blend ──────────────────────────────────────
            soft_mask = feather_mask(canvas_mask, blur_radius=25)
            soft_mask = (soft_mask.astype(float) * blend_strength).clip(0, 255).astype(np.uint8)
            result    = alpha_blend(canvas_fg, target_bgr, soft_mask)

            # ── 7. Seamless clone (best-effort) ───────────────────────────────
            try:
                cx = int((dst_x1 + dst_x2) / 2)
                cy = int((dst_y1 + dst_y2) / 2)
                sc_mask = (canvas_mask > 10).astype(np.uint8) * 255
                result  = cv2.seamlessClone(
                    canvas_fg, target_bgr, sc_mask,
                    (cx, cy), cv2.NORMAL_CLONE,
                )
            except Exception as e:
                print(f"[BodySwapper] seamlessClone skipped: {e}")

            return result, "Body swap completed successfully."

        except Exception as exc:
            return None, f"Body swap error: {exc}"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _segment(self, bgr: np.ndarray) -> np.ndarray:
        """Return a uint8 single-channel person mask via rembg u2net_human_seg."""
        from rembg import remove
        pil    = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = remove(pil, only_mask=True, session=self._get_session())
        mask   = np.array(result)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask
