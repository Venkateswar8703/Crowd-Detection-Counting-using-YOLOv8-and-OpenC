"""
Crowd Detection & Counting using YOLOv8 and OpenCV
====================================================
CLI entry point — supports webcam, video file, and image inputs.

Usage
-----
    python main.py --source webcam
    python main.py --source video --input data/crowd_video.mp4
    python main.py --source image --input data/crowd_image.jpg
"""

import argparse
import sys
import time

import cv2

from config.settings import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MODEL_PATH, WINDOW_NAME
from src.detector import CrowdDetector
from src.logger import DetectionLogger
from src.visualizer import Visualizer
from utils.helpers import FPSCounter, create_video_writer, resize_frame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Crowd Detection & Counting using YOLOv8 and OpenCV",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        choices=["webcam", "video", "image"],
        help="Input source type (default: webcam)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input video or image file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"YOLOv8 model path or name (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Detection confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
        help=f"NMS IoU threshold (default: {IOU_THRESHOLD})",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to outputs/ directory",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Export per-frame CSV detection log",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without GUI display (headless mode)",
    )
    return parser.parse_args()


def process_video(args: argparse.Namespace, detector: CrowdDetector) -> None:
    """Process a video stream (webcam or file).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    detector : CrowdDetector
        Initialized YOLOv8 detector instance.
    """
    # Open video source
    if args.source == "webcam":
        cap = cv2.VideoCapture(0)
        print("[INFO] Starting webcam stream...")
    else:
        if not args.input:
            print("[ERROR] --input path required for video source.")
            sys.exit(1)
        cap = cv2.VideoCapture(args.input)
        print(f"[INFO] Opening video: {args.input}")

    if not cap.isOpened():
        print("[ERROR] Failed to open video source.")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"[INFO] Resolution: {frame_w}x{frame_h} | Source FPS: {source_fps:.1f}")

    # Optional: video writer
    writer = None
    if args.save:
        writer = create_video_writer("crowd_detection_output", frame_w, frame_h, source_fps)

    # Optional: CSV logger
    logger = None
    if args.log:
        logger = DetectionLogger()

    fps_counter = FPSCounter()
    frame_id = 0

    print("[INFO] Processing... Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n[INFO] End of video stream.")
            break

        frame_id += 1
        timestamp = frame_id / source_fps

        # ── Detection ────────────────────────────────────────
        result = detector.detect(frame)

        # ── Visualization ────────────────────────────────────
        fps = fps_counter.tick()
        frame = Visualizer.draw_detections(frame, result)
        frame = Visualizer.draw_hud(frame, result.person_count, fps, result.avg_confidence)

        # ── Logging ──────────────────────────────────────────
        if logger:
            logger.log(frame_id, timestamp, result)

        # ── Output ───────────────────────────────────────────
        if writer:
            writer.write(frame)

        if not args.no_display:
            display_frame = resize_frame(frame)
            cv2.imshow(WINDOW_NAME, display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] User quit.")
                break

        # Console feedback every 100 frames
        if frame_id % 100 == 0:
            print(
                f"  Frame {frame_id:>6d} | "
                f"People: {result.person_count:>3d} | "
                f"FPS: {fps:.1f}"
            )

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if logger:
        logger.close()
    if not args.no_display:
        cv2.destroyAllWindows()

    print("[INFO] Video processing complete.")


def process_image(args: argparse.Namespace, detector: CrowdDetector) -> None:
    """Process a single static image.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    detector : CrowdDetector
        Initialized YOLOv8 detector instance.
    """
    if not args.input:
        print("[ERROR] --input path required for image source.")
        sys.exit(1)

    frame = cv2.imread(args.input)
    if frame is None:
        print(f"[ERROR] Could not read image: {args.input}")
        sys.exit(1)

    print(f"[INFO] Processing image: {args.input}")

    # ── Detection ────────────────────────────────────────────
    start = time.perf_counter()
    result = detector.detect(frame)
    elapsed = time.perf_counter() - start

    # ── Visualization ────────────────────────────────────────
    frame = Visualizer.draw_detections(frame, result)
    frame = Visualizer.draw_hud(
        frame,
        result.person_count,
        fps=1.0 / elapsed if elapsed > 0 else 0.0,
        avg_confidence=result.avg_confidence,
    )

    print(f"[RESULT] People detected: {result.person_count}")
    print(f"[RESULT] Avg confidence:  {result.avg_confidence:.1%}")
    print(f"[RESULT] Inference time:  {elapsed * 1000:.1f} ms")

    # ── Save ─────────────────────────────────────────────────
    if args.save:
        import os
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", "crowd_detection_output.jpg")
        cv2.imwrite(out_path, frame)
        print(f"[INFO] Saved annotated image: {out_path}")

    # ── Display ──────────────────────────────────────────────
    if not args.no_display:
        display_frame = resize_frame(frame)
        cv2.imshow(WINDOW_NAME, display_frame)
        print("[INFO] Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    """Application entry point."""
    args = parse_args()

    print("=" * 60)
    print("  Crowd Detection & Counting using YOLOv8 + OpenCV")
    print("=" * 60)
    print(f"  Source     : {args.source}")
    print(f"  Model      : {args.model}")
    print(f"  Confidence : {args.confidence}")
    print(f"  IoU        : {args.iou}")
    print(f"  Save       : {args.save}")
    print(f"  Log        : {args.log}")
    print("=" * 60 + "\n")

    # Initialize detector
    detector = CrowdDetector(
        model_path=args.model,
        confidence=args.confidence,
        iou=args.iou,
    )

    # Route to appropriate handler
    if args.source in ("webcam", "video"):
        process_video(args, detector)
    else:
        process_image(args, detector)


if __name__ == "__main__":
    main()
