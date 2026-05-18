"""
Crowd Detection & Counting — Helper Utilities
===============================================
FPS counter, video writer factory, and frame resizing.
"""

import os
import time

import cv2
import numpy as np

from config.settings import MAX_DISPLAY_WIDTH, OUTPUT_DIR, OUTPUT_VIDEO_CODEC, OUTPUT_VIDEO_FPS


class FPSCounter:
    """Lightweight rolling FPS calculator."""

    def __init__(self, smoothing: int = 30) -> None:
        self._smoothing = smoothing
        self._timestamps: list[float] = []

    def tick(self) -> float:
        """Record a frame timestamp and return smoothed FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)

        # Keep only the last N timestamps
        if len(self._timestamps) > self._smoothing:
            self._timestamps = self._timestamps[-self._smoothing:]

        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def create_video_writer(
    output_name: str,
    frame_width: int,
    frame_height: int,
    fps: float = OUTPUT_VIDEO_FPS,
) -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter for saving annotated output.

    Parameters
    ----------
    output_name : str
        Base filename (without extension).
    frame_width : int
        Width of output frames.
    frame_height : int
        Height of output frames.
    fps : float
        Output video frame rate.

    Returns
    -------
    cv2.VideoWriter
        Configured video writer instance.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"{output_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
    writer = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))
    print(f"[INFO] Saving output video to: {filepath}")
    return writer


def resize_frame(frame: np.ndarray, max_width: int = MAX_DISPLAY_WIDTH) -> np.ndarray:
    """Resize a frame for display if it exceeds max_width.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR frame.
    max_width : int
        Maximum allowed display width in pixels.

    Returns
    -------
    np.ndarray
        Resized (or original) frame.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
