"""
Crowd Detection & Counting — CSV Detection Logger
===================================================
Exports per-frame detection metadata to a CSV file
for downstream analysis and reporting.
"""

import csv
import os
from typing import Optional

from config.settings import CSV_LOG_FILENAME, OUTPUT_DIR
from src.detector import FrameResult


class DetectionLogger:
    """Buffers and writes per-frame detection stats to CSV.

    Parameters
    ----------
    output_dir : str
        Directory to save the CSV log file.
    filename : str
        Name of the CSV output file.
    """

    FIELDNAMES = [
        "frame_id",
        "timestamp_sec",
        "person_count",
        "avg_confidence",
        "detections",
    ]

    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        filename: str = CSV_LOG_FILENAME,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, filename)
        self._file = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        print(f"[INFO] Logging detections to: {self.filepath}")

    def log(
        self,
        frame_id: int,
        timestamp: float,
        result: FrameResult,
    ) -> None:
        """Write a single frame's detection results.

        Parameters
        ----------
        frame_id : int
            Sequential frame number.
        timestamp : float
            Time offset in seconds from video start.
        result : FrameResult
            Detection results for this frame.
        """
        det_str = "; ".join(
            f"[{d.bbox[0]},{d.bbox[1]},{d.bbox[2]},{d.bbox[3]}] conf={d.confidence}"
            for d in result.detections
        )

        self._writer.writerow(
            {
                "frame_id": frame_id,
                "timestamp_sec": round(timestamp, 3),
                "person_count": result.person_count,
                "avg_confidence": result.avg_confidence,
                "detections": det_str,
            }
        )

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file and not self._file.closed:
            self._file.close()
            print(f"[INFO] Detection log saved: {self.filepath}")
