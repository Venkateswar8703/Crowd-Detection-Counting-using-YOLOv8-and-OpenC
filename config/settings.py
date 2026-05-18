"""
Crowd Detection & Counting — Central Configuration
====================================================
All tunable parameters for the detection pipeline.
"""

# ── Model Configuration ──────────────────────────────────────
MODEL_PATH = "yolov8n.pt"           # Ultralytics model variant
TARGET_CLASS_ID = 0                  # COCO class index for 'person'
TARGET_CLASS_NAME = "person"

# ── Detection Thresholds ─────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35          # Minimum confidence to accept a detection
IOU_THRESHOLD = 0.45                 # IoU threshold for Non-Maximum Suppression
INPUT_SIZE = 640                     # Model input resolution (pixels)

# ── Visualization ────────────────────────────────────────────
BBOX_COLOR = (0, 255, 0)            # Bounding box color (BGR: green)
BBOX_THICKNESS = 2                   # Bounding box line width
TEXT_COLOR = (255, 255, 255)         # Label text color (BGR: white)
TEXT_BG_COLOR = (0, 180, 0)         # Label background color
TEXT_FONT_SCALE = 0.5                # Font scale for labels
HUD_BG_COLOR = (40, 40, 40)         # Heads-up display background
HUD_TEXT_COLOR = (0, 255, 255)       # HUD text color (BGR: cyan)
HUD_FONT_SCALE = 0.8                # Font scale for HUD elements
HUD_MARGIN = 10                      # Pixel margin inside HUD panel

# ── Output ───────────────────────────────────────────────────
OUTPUT_DIR = "outputs"               # Directory for saved results
OUTPUT_VIDEO_FPS = 25.0              # FPS for saved video files
OUTPUT_VIDEO_CODEC = "mp4v"          # FourCC codec for video writer
CSV_LOG_FILENAME = "detection_log.csv"

# ── Display ──────────────────────────────────────────────────
WINDOW_NAME = "Crowd Detection - YOLOv8"
MAX_DISPLAY_WIDTH = 1280             # Resize display window if frame exceeds this
