"""
Video processor — extracts frames from an input video, applies face or body
swap to each frame, then re-encodes the result with FFmpeg (audio preserved).

Speed optimisations
-------------------
* Source face is detected **once** before the loop (never per-frame).
* Target face detection is cached and reused for DET_INTERVAL frames — faces
  don't move much between consecutive frames at normal frame rates.
* Video frames are capped at 720p for processing (upscaled back for writing).
* A hard cap of MAX_FRAMES is enforced to keep processing times reasonable on
  free CPU tiers.
"""

import cv2
import os
import tempfile
import numpy as np
from pathlib import Path

MAX_FRAMES   = 600   # ~20 s at 30 fps
DET_INTERVAL = 5     # re-detect target faces every N frames


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
        fast_mode: bool = False,     # skip every other frame (~2x speed)
        start_frame: int = 0,        # resume from this frame index
        progress=None,
    ) -> tuple[str | None, str]:
        """
        Process every frame of *video_path*, applying the selected swap mode.
        Set *start_frame* > 0 to resume after a dropped connection.
        Partial output is always saved — even if processing is interrupted.

        Returns:
            (output_path, status_message)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Could not open video file."

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Clamp start_frame
        start_frame = max(0, min(start_frame, total_frames - 1))
        remaining   = total_frames - start_frame

        if remaining > MAX_FRAMES:
            cap.release()
            return None, (
                f"Segment starting at frame {start_frame} has {remaining} frames — "
                f"maximum allowed is {MAX_FRAMES} (~{MAX_FRAMES / fps:.0f} s at {fps:.0f} fps). "
                "Increase the start frame or trim the video."
            )

        # ── Pre-compute source face once (big win for face-swap mode) ─────────
        source_face = None
        if mode == "face" and self.face_swapper:
            source_face = self.face_swapper.get_source_face(source_bgr)
            if source_face is None:
                cap.release()
                return None, "No face detected in source image."

        # ── Seek to start_frame — use FFmpeg cut for instant seek ──────────────
        # cap.set(POS_FRAMES) is slow: OpenCV decodes every frame up to the
        # target.  FFmpeg keyframe-seeks in milliseconds.
        segment_path = None
        if start_frame > 0:
            start_time = start_frame / fps
            segment_path = tempfile.mktemp(suffix="_segment.mp4")
            try:
                import ffmpeg as _ffmpeg
                (
                    _ffmpeg.input(video_path, ss=start_time)
                    .output(segment_path, c="copy", avoid_negative_ts="make_zero")
                    .overwrite_output()
                    .run(quiet=True)
                )
                cap.release()
                cap = cv2.VideoCapture(segment_path)
                print(f"[VideoProcessor] Resumed via FFmpeg cut at frame {start_frame} ({start_time:.2f}s)")
            except Exception as e:
                print(f"[VideoProcessor] FFmpeg seek failed ({e}), falling back to slow seek")
                segment_path = None
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Use AVI + XVID for the intermediate file — far more reliable than
        # mp4v on Linux (HF Spaces).  FFmpeg converts it to H.264/mp4 after.
        raw_out_path = tempfile.mktemp(suffix="_raw.avi")
        fourcc       = cv2.VideoWriter_fourcc(*"XVID")
        writer       = cv2.VideoWriter(raw_out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            # XVID not available — fall back to MJPG
            raw_out_path = tempfile.mktemp(suffix="_raw.avi")
            fourcc  = cv2.VideoWriter_fourcc(*"MJPG")
            writer  = cv2.VideoWriter(raw_out_path, fourcc, fps, (width, height))

        frame_idx        = start_frame   # absolute frame number in the source video
        processed        = 0
        errors           = 0
        cached_tgt_faces = None
        last_result      = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if progress is not None and total_frames > 0:
                    progress(
                        (frame_idx - start_frame) / remaining,
                        f"Frame {frame_idx + 1} / {total_frames}  "
                        f"(resume at {frame_idx} if interrupted)",
                    )

                # Fast mode: skip odd frames — write the ORIGINAL frame (not a
                # duplicate) so motion stays smooth with no stutter or blur.
                # Only applies to face swap; body swap needs every frame.
                if fast_mode and mode == "face" and (frame_idx - start_frame) % 2 == 1:
                    writer.write(frame)   # original frame keeps motion fluid
                    frame_idx += 1
                    continue

                # Only re-detect target faces every DET_INTERVAL frames
                use_cache = (mode == "face") and (frame_idx % DET_INTERVAL != 0) and (cached_tgt_faces is not None)

                result_frame, new_faces = self._process_frame(
                    source_bgr, frame, mode, enhance, blend_strength,
                    source_face=source_face,
                    cached_target_faces=cached_tgt_faces if use_cache else None,
                )

                if mode == "face" and new_faces is not None:
                    cached_tgt_faces = new_faces if new_faces else cached_tgt_faces

                if result_frame is not None:
                    writer.write(result_frame)
                    last_result = result_frame
                    processed += 1
                else:
                    writer.write(frame)
                    last_result = frame
                    errors += 1

                frame_idx += 1

        except Exception as loop_err:
            print(f"[VideoProcessor] Loop interrupted at frame {frame_idx}: {loop_err}")

        finally:
            cap.release()
            writer.release()
            if segment_path:
                try:
                    os.unlink(segment_path)
                except OSError:
                    pass

        frames_done = frame_idx - start_frame
        if frames_done == 0:
            try:
                os.unlink(raw_out_path)
            except OSError:
                pass
            return None, f"No frames processed. Try resuming from frame {start_frame}."

        # Re-encode with H.264 and merge original audio via FFmpeg
        # Pass start_time so audio lines up with the resumed segment
        start_time = start_frame / fps
        final_path = self._ffmpeg_encode(video_path, raw_out_path, audio_start=start_time)

        try:
            os.unlink(raw_out_path)
        except OSError:
            pass

        partial = frames_done < remaining
        status = (
            f"{'Partial — ' if partial else ''}Frames {start_frame}–{frame_idx - 1} "
            f"({processed} swapped{', ' + str(errors) + ' skipped' if errors else ''}). "
            + (f"Resume from frame {frame_idx} to continue." if partial else "Done.")
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
        source_face=None,
        cached_target_faces=None,
    ):
        """Returns (result_frame_or_None, detected_faces_or_None)."""
        try:
            if mode == "face" and self.face_swapper:
                result, faces = self.face_swapper.swap_frame(
                    frame,
                    source_face,
                    cached_target_faces=cached_target_faces,
                    enhance=enhance,
                )
                return result, faces
            elif mode == "body" and self.body_swapper:
                result, _ = self.body_swapper.swap(
                    source_bgr, frame, blend_strength=blend_strength
                )
                return result, None
        except Exception as e:
            print(f"[VideoProcessor] Frame error: {e}")
        return None, None

    @staticmethod
    def _ffmpeg_encode(original_video_path: str, processed_raw_path: str, audio_start: float = 0.0) -> str:
        """
        Re-encode processed frames as H.264 mp4 and merge the original audio.
        audio_start: seconds into the original audio (for resumed segments).
        Returns the output path; raises if encoding fails so caller can report it.
        """
        final_path = tempfile.mktemp(suffix="_output.mp4")
        try:
            import ffmpeg
            import subprocess

            video_in = ffmpeg.input(processed_raw_path)
            audio_in = ffmpeg.input(original_video_path)

            # Build output streams
            streams = [video_in.video]
            # Only attach audio if the source has an audio track
            try:
                probe = ffmpeg.probe(original_video_path)
                has_audio = any(s["codec_type"] == "audio" for s in probe["streams"])
            except Exception:
                has_audio = False

            if has_audio:
                if audio_start > 0:
                    audio_in = ffmpeg.input(original_video_path, ss=audio_start)
                streams.append(audio_in.audio)

            out_kwargs = dict(
                vcodec="libx264",
                crf=18,              # 18 = visually lossless (was 23)
                preset="fast",
                pix_fmt="yuv420p",   # widest player compatibility
                **{"vf": "unsharp=5:5:1.0:5:5:0.0"},  # mild luma sharpening
            )
            if has_audio:
                out_kwargs.update(acodec="aac", audio_bitrate="192k")

            (
                ffmpeg.output(*streams, final_path, **out_kwargs)
                .overwrite_output()
                .run(quiet=False, capture_stdout=True, capture_stderr=True)
            )

            # Validate output
            if not os.path.exists(final_path) or os.path.getsize(final_path) < 1024:
                raise RuntimeError("FFmpeg produced an empty output file.")

            return final_path

        except ffmpeg.Error as e:
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            print(f"[VideoProcessor] FFmpeg error:\n{stderr}")
            # Return the raw file as fallback so the user gets something
            return processed_raw_path
        except Exception as e:
            print(f"[VideoProcessor] FFmpeg encode failed: {e}")
            return processed_raw_path
