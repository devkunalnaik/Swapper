"""
Face & Body Swapper — Gradio app for Hugging Face Spaces (CPU).

Tabs
----
1. Face Swap  (Image)
2. Body Swap  (Image)
3. Face Swap  (Video)
4. Body Swap  (Video)
"""

import cv2
import numpy as np
import gradio as gr

from utils.image_utils import pil_to_bgr, bgr_to_pil

# ── Lazy processor singletons ─────────────────────────────────────────────────
_face_swapper = None
_body_swapper = None


def _get_face_swapper():
    global _face_swapper
    if _face_swapper is None:
        from processors.face_swap import FaceSwapper
        _face_swapper = FaceSwapper()
    return _face_swapper


def _get_body_swapper():
    global _body_swapper
    if _body_swapper is None:
        from processors.body_swap import BodySwapper
        _body_swapper = BodySwapper()
    return _body_swapper


# ── Processing functions ─────────────────────────────────────────────────────

def face_swap_image(source_img, target_img, enhance):
    if source_img is None or target_img is None:
        return None, "Please upload both a source and a target image."
    try:
        result, msg = _get_face_swapper().swap(
            pil_to_bgr(source_img),
            pil_to_bgr(target_img),
            enhance=enhance,
        )
        return (bgr_to_pil(result) if result is not None else None), msg
    except Exception as e:
        return None, f"Error: {e}"


def body_swap_image(source_img, target_img, blend_strength):
    if source_img is None or target_img is None:
        return None, "Please upload both a source and a target image."
    try:
        result, msg = _get_body_swapper().swap(
            pil_to_bgr(source_img),
            pil_to_bgr(target_img),
            blend_strength=blend_strength,
        )
        return (bgr_to_pil(result) if result is not None else None), msg
    except Exception as e:
        return None, f"Error: {e}"


def face_swap_video(source_img, target_video, enhance, progress=gr.Progress()):
    if source_img is None or target_video is None:
        return None, "Please upload a source face image and a target video."

    from processors.video_processor import VideoProcessor
    processor = VideoProcessor(face_swapper=_get_face_swapper())
    output_path, msg = processor.process_video(
        pil_to_bgr(source_img),
        target_video,
        mode="face",
        enhance=enhance,
        progress=progress,
    )
    return output_path, msg


def body_swap_video(source_img, target_video, blend_strength, progress=gr.Progress()):
    if source_img is None or target_video is None:
        return None, "Please upload a source body image and a target video."

    from processors.video_processor import VideoProcessor
    processor = VideoProcessor(body_swapper=_get_body_swapper())
    output_path, msg = processor.process_video(
        pil_to_bgr(source_img),
        target_video,
        mode="body",
        blend_strength=blend_strength,
        progress=progress,
    )
    return output_path, msg


# ── Gradio UI ─────────────────────────────────────────────────────────────────

DISCLAIMER = """
> ⚠️ **Ethical Use Only** — Only process images/videos of yourself or with
> explicit written consent of every person depicted.  Do **not** use this tool
> to create misleading, deceptive, non-consensual, or harmful content.
"""

with gr.Blocks(title="Face & Body Swapper", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔄 Face & Body Swapper")
    gr.Markdown(DISCLAIMER)

    with gr.Tabs():

        # ── Tab 1: Face Swap — Image ──────────────────────────────────────────
        with gr.Tab("Face Swap · Image"):
            gr.Markdown(
                "The **source** face is placed onto every detected face in the **target** image."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    fi_source = gr.Image(label="Source — face to use", type="pil")
                    fi_target = gr.Image(label="Target — image to modify", type="pil")
                    fi_enhance = gr.Checkbox(
                        label="Enhance face quality (sharpening + contrast)",
                        value=True,
                    )
                    fi_btn = gr.Button("Swap Faces", variant="primary")
                with gr.Column(scale=1):
                    fi_output = gr.Image(label="Result", type="pil")
                    fi_status = gr.Textbox(label="Status", interactive=False)

            fi_btn.click(
                face_swap_image,
                inputs=[fi_source, fi_target, fi_enhance],
                outputs=[fi_output, fi_status],
            )

        # ── Tab 2: Body Swap — Image ──────────────────────────────────────────
        with gr.Tab("Body Swap · Image"):
            gr.Markdown(
                "The **source** person's body is transplanted into the **target** scene."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    bi_source = gr.Image(label="Source — body to use", type="pil")
                    bi_target = gr.Image(label="Target — scene to modify", type="pil")
                    bi_blend = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.85, step=0.05,
                        label="Blend Strength",
                    )
                    bi_btn = gr.Button("Swap Body", variant="primary")
                with gr.Column(scale=1):
                    bi_output = gr.Image(label="Result", type="pil")
                    bi_status = gr.Textbox(label="Status", interactive=False)

            bi_btn.click(
                body_swap_image,
                inputs=[bi_source, bi_target, bi_blend],
                outputs=[bi_output, bi_status],
            )

        # ── Tab 3: Face Swap — Video ──────────────────────────────────────────
        with gr.Tab("Face Swap · Video"):
            gr.Markdown(
                "Applies face swap to every frame of the **target** video.  "
                "Max video length: **600 frames** (~20 s at 30 fps)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    fv_source = gr.Image(label="Source Face Image", type="pil")
                    fv_target = gr.Video(label="Target Video")
                    fv_enhance = gr.Checkbox(
                        label="Enhance faces (sharpening + contrast)",
                        value=True,
                    )
                    fv_btn = gr.Button("Swap Faces in Video", variant="primary")
                with gr.Column(scale=1):
                    fv_output = gr.Video(label="Result Video")
                    fv_status = gr.Textbox(label="Status", interactive=False)

            fv_btn.click(
                face_swap_video,
                inputs=[fv_source, fv_target, fv_enhance],
                outputs=[fv_output, fv_status],
            )

        # ── Tab 4: Body Swap — Video ──────────────────────────────────────────
        with gr.Tab("Body Swap · Video"):
            gr.Markdown(
                "Applies body swap to every frame of the **target** video.  "
                "Max video length: **600 frames** (~20 s at 30 fps)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    bv_source = gr.Image(label="Source Body Image", type="pil")
                    bv_target = gr.Video(label="Target Video")
                    bv_blend = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.85, step=0.05,
                        label="Blend Strength",
                    )
                    bv_btn = gr.Button("Swap Body in Video", variant="primary")
                with gr.Column(scale=1):
                    bv_output = gr.Video(label="Result Video")
                    bv_status = gr.Textbox(label="Status", interactive=False)

            bv_btn.click(
                body_swap_video,
                inputs=[bv_source, bv_target, bv_blend],
                outputs=[bv_output, bv_status],
            )

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(ssr_mode=False, show_api=False)
