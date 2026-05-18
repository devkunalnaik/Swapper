"""
Video processor — extracts frames from an input video, applies face or body
swap to each frame, then re-encodes the result with FFmpeg (audio preserved).

A hard cap of MAX_FRAMES is enforced to keep processing times reasonable on
free GPU tiers.
"""

import cv2
import os
import tempfile
import numpy as np
from pathlib import Path

MAX_FRAMES = 600  # ~20 s at 30 fps — raise for paid/GPU tiers


class VideoProcessor:
    def __init__(
        self,
        face_swapper=None,
        body_swapper=None,
    ):
        self.face_swapper = face_swapper
        self.body_swapper = body_swapper

    # ── Public API ────────────────────────────────────────────────────────────

    def process_video(
        self,
        source_bgr: np.ndarray,
        video_path: str,
        mode: str = "face",          # "face" | "body"
        enhance: bool = False,
        blend_strength: float = 0.85,
        progress=None,
    ) -> tuple[str | None, str]:
        """
        Process every frame of *video_path*, applying the selected swap mode.

        Returns:
            (output_path, status_message)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Could not open video file."

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > MAX_FRAMES:
            cap.release()
            return None, (
                f"Video has {total_frames} frames — maximum allowed is "
                f"{MAX_FRAMES} (~{MAX_FRAMES / fps:.0f} s at {fps:.0f} fps). "
                "Please trim the video and try again."
            )

        # Temp file for raw processed frames (mp4v codec)
        raw_out_path = tempfile.mktemp(suffix="_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_out_path, fourcc, fps, (width, height))

        frame_idx = 0
        processed = 0
        errors = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if progress is not None and total_frames > 0:
                progress(
                    frame_idx / total_frames,
                    f"Processing frame {frame_idx + 1} / {total_frames}",
                )

            result_frame = self._process_frame(
                source_bgr, frame, mode, enhance, blend_strength
            )

            if result_frame is not None:
                writer.write(result_frame)
                processed += 1
            else:
                writer.write(frame)  # keep original on failure
                errors += 1

            frame_idx += 1

        cap.release()
        writer.release()

        # Re-encode with H.264 and merge original audio via FFmpeg
        final_path = self._ffmpeg_encode(video_path, raw_out_path)

        # Clean up raw file
        try:
            os.unlink(raw_out_path)
        except OSError:
            pass

        status = (
            f"Done — {processed}/{frame_idx} frames processed"
            + (f" ({errors} skipped)" if errors else "")
            + "."
        )
        return final_path, status

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _process_frame(
        self,
        source_bgr: np.ndarray,
        frame: np.ndarray,
        mode: str,
        enhance: bool,
        blend_strength: float,
    ) -> np.ndarray | None:
        try:
            if mode == "face" and self.face_swapper:
                result, _ = self.face_swapper.swap(source_bgr, frame, enhance=enhance)
                return result
            elif mode == "body" and self.body_swapper:
                result, _ = self.body_swapper.swap(
                    source_bgr, frame, blend_strength=blend_strength
                )
                return result
        except Exception as e:
            print(f"[VideoProcessor] Frame error: {e}")
        return None

    @staticmethod
    def _ffmpeg_encode(original_video_path: str, processed_raw_path: str) -> str:
        """
        Re-encode processed frames as H.264 and copy the original audio track.
        Falls back to the raw file if FFmpeg is unavailable.
        """
        final_path = tempfile.mktemp(suffix="_output.mp4")
        try:
            import ffmpeg

            video_stream = ffmpeg.input(processed_raw_path).video
            audio_stream = ffmpeg.input(original_video_path).audio

            (
                ffmpeg.output(
                    video_stream,
                    audio_stream,
                    final_path,
                    vcodec="libx264",
                    crf=23,
                    preset="fast",
                    acodec="aac",
                    audio_bitrate="192k",
                )
                .overwrite_output()
                .run(quiet=True)
            )
            return final_path

        except Exception as e:
            print(f"[VideoProcessor] FFmpeg re-encode failed ({e}), returning raw file.")
            return processed_raw_path
