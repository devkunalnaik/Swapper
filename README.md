---
title: Face & Body Swapper
emoji: 🔄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.6.0
app_file: app.py
pinned: false
license: mit
---

# Face & Body Swapper

A tool for swapping faces and bodies in images and videos using AI.

## Features
- **Face Swap (Image)** — Replace faces in a target image with a source face
- **Body Swap (Image)** — Transplant a person's body onto a target scene
- **Face Swap (Video)** — Apply face swap frame-by-frame on a video
- **Body Swap (Video)** — Apply body swap frame-by-frame on a video

## Tech Stack
- [InsightFace](https://github.com/deepinsight/insightface) — Face detection & swapping
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Face enhancement
- [rembg](https://github.com/danielgatis/rembg) — Person segmentation
- [MediaPipe](https://developers.google.com/mediapipe) — Pose estimation
- [OpenCV](https://opencv.org/) — Image/video processing
- [FFmpeg](https://ffmpeg.org/) — Video encoding

## Ethical Use

> ⚠️ **Important**: Only use with explicit consent of all persons involved.
> Do not use to create misleading, deceptive, non-consensual, or harmful content.
> The creators of this tool are not responsible for misuse.
