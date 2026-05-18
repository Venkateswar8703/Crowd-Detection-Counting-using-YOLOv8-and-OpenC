"""
Crowd Detection & Counting — YOLOv8 Detection Engine
=====================================================
Wraps the Ultralytics YOLOv8 model for person-class detection
with configurable confidence and NMS thresholds.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from config.settings import (
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    MODEL_PATH,
    TARGET_CLASS_ID,
)


@dataclass
class Detection:
    """Single detected person bounding box."""

    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    class_id: int


@dataclass
class FrameResult:
    """Aggregated detection results for a single frame."""

    detections: List[Detection] = field(default_factory=list)
    person_count: int = 0
    avg_confidence: float = 0.0


class CrowdDetector:
    """YOLOv8-based crowd (person) detector.

    Parameters
    ----------
    model_path : str
        Path or name of the YOLOv8 model weights.
    confidence : float
        Minimum detection confidence threshold.
    iou : float
        IoU threshold for Non-Maximum Suppression.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        confidence: float = CONFIDENCE_THRESHOLD,
        iou: float = IOU_THRESHOLD,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        print(f"[INFO] Loaded model: {model_path}")

    def detect(self, frame: np.ndarray) -> FrameResult:
        """Run detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Input image in BGR format (OpenCV default).

        Returns
        -------
        FrameResult
            Detected persons with bounding boxes and metadata.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=[TARGET_CLASS_ID],
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=round(conf, 3),
                        class_id=cls_id,
                    )
                )

        person_count = len(detections)
        avg_conf = (
            round(sum(d.confidence for d in detections) / person_count, 3)
            if person_count > 0
            else 0.0
        )

        return FrameResult(
            detections=detections,
            person_count=person_count,
            avg_confidence=avg_conf,
        )
