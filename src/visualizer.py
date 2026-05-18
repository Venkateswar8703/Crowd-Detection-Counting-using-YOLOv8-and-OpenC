"""
Crowd Detection & Counting — Visualization Module
===================================================
Draws bounding boxes, confidence labels, and a HUD panel
onto OpenCV frames.
"""

import cv2
import numpy as np

from config.settings import (
    BBOX_COLOR,
    BBOX_THICKNESS,
    HUD_BG_COLOR,
    HUD_FONT_SCALE,
    HUD_MARGIN,
    HUD_TEXT_COLOR,
    TEXT_BG_COLOR,
    TEXT_COLOR,
    TEXT_FONT_SCALE,
)
from src.detector import FrameResult


class Visualizer:
    """Overlay detection annotations and a heads-up display on frames."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def draw_detections(frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw bounding boxes and per-detection confidence labels.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to annotate (modified in-place).
        result : FrameResult
            Detection results for the current frame.

        Returns
        -------
        np.ndarray
            Annotated frame.
        """
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

            # Confidence label
            label = f"Person {det.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label, Visualizer.FONT, TEXT_FONT_SCALE, 1
            )
            cv2.rectangle(
                frame, (x1, y1 - th - baseline - 4), (x1 + tw, y1), TEXT_BG_COLOR, -1
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 2),
                Visualizer.FONT,
                TEXT_FONT_SCALE,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

        return frame

    @staticmethod
    def draw_hud(
        frame: np.ndarray,
        person_count: int,
        fps: float,
        avg_confidence: float,
    ) -> np.ndarray:
        """Render a translucent heads-up display panel at the top-left.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to annotate.
        person_count : int
            Number of detected persons.
        fps : float
            Current processing speed.
        avg_confidence : float
            Mean confidence across detections.

        Returns
        -------
        np.ndarray
            Frame with HUD overlay.
        """
        lines = [
            f"People Count: {person_count}",
            f"FPS: {fps:.1f}",
            f"Avg Confidence: {avg_confidence:.1%}",
        ]

        # Calculate HUD panel dimensions
        max_tw = 0
        total_th = 0
        line_heights = []
        for line in lines:
            (tw, th), baseline = cv2.getTextSize(
                line, Visualizer.FONT, HUD_FONT_SCALE, 2
            )
            max_tw = max(max_tw, tw)
            total_th += th + baseline + 8
            line_heights.append((th, baseline))

        panel_w = max_tw + 2 * HUD_MARGIN
        panel_h = total_th + 2 * HUD_MARGIN

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), HUD_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text lines
        y_offset = HUD_MARGIN
        for i, line in enumerate(lines):
            th, baseline = line_heights[i]
            y_offset += th + 4
            cv2.putText(
                frame,
                line,
                (HUD_MARGIN, y_offset),
                Visualizer.FONT,
                HUD_FONT_SCALE,
                HUD_TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
            y_offset += baseline + 4

        return frame
