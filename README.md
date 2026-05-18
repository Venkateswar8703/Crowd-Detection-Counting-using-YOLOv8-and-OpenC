# 🎯 Crowd Detection & Counting using YOLOv8 and OpenCV

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A real-time crowd detection and people-counting system powered by **YOLOv8** object detection and **OpenCV** video processing. The pipeline supports live webcam feeds, video files, and static images — producing annotated outputs with bounding boxes, confidence scores, and a running head count.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Results](#-results)
- [Technical Details](#-technical-details)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Real-Time Detection** | Process live webcam streams at 25-30+ FPS |
| **Video Analysis** | Batch-process pre-recorded video files with annotated output |
| **Image Detection** | Single-frame crowd analysis on static images |
| **Adaptive Confidence** | Configurable confidence & IoU thresholds to balance precision/recall |
| **Non-Max Suppression** | Built-in NMS to eliminate duplicate detections |
| **Live HUD** | On-screen head count, FPS counter, and detection metadata |
| **CSV Logging** | Frame-by-frame detection logs exported to CSV for downstream analysis |
| **Modular Design** | Clean separation of detection, visualization, and I/O logic |

---

## 🏗 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   Input Source   │────▶│  YOLOv8 Detector │────▶│   Post-Processor  │
│ (webcam/video/   │     │  (Ultralytics)   │     │  (NMS + Filtering)│
│  image)          │     └──────────────────┘     └───────────────────┘
└─────────────────┘                                        │
                                                           ▼
                                               ┌───────────────────┐
                                               │   Visualizer &    │
                                               │   CSV Logger      │
                                               │  (OpenCV + CSV)   │
                                               └───────────────────┘
                                                           │
                                                           ▼
                                               ┌───────────────────┐
                                               │   Output          │
                                               │ (Display/Save)    │
                                               └───────────────────┘
```

---

## 📁 Project Structure

```
Crowd-Detection-Counting-using-YOLOv8-and-OpenCv/
│
├── config/
│   └── settings.py            # Central configuration (thresholds, paths, colors)
│
├── src/
│   ├── __init__.py
│   ├── detector.py            # YOLOv8 detection engine
│   ├── visualizer.py          # Bounding-box drawing & HUD overlay
│   └── logger.py              # CSV frame-log exporter
│
├── utils/
│   ├── __init__.py
│   └── helpers.py             # FPS counter, video-writer factory, utilities
│
├── outputs/                   # Generated annotated videos & logs (git-ignored)
│
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.8+**
- pip package manager
- (Optional) NVIDIA GPU with CUDA for accelerated inference

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Venkateswar8703/Crowd-Detection-Counting-using-YOLOv8-and-OpenC.git
cd Crowd-Detection-Counting-using-YOLOv8-and-OpenC

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The YOLOv8 model weights (`yolov8n.pt`) are downloaded automatically on first run via the Ultralytics hub.

---

## 🚀 Usage

### 1. Webcam (Real-Time)

```bash
python main.py --source webcam
```

### 2. Video File

```bash
python main.py --source video --input data/crowd_video.mp4
```

### 3. Static Image

```bash
python main.py --source image --input data/crowd_image.jpg
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `webcam` | Input source: `webcam`, `video`, or `image` |
| `--input` | `None` | Path to video or image file |
| `--model` | `yolov8n.pt` | YOLOv8 model variant (`yolov8n/s/m/l/x.pt`) |
| `--confidence` | `0.35` | Minimum detection confidence |
| `--iou` | `0.45` | IoU threshold for NMS |
| `--save` | `False` | Save annotated output to `outputs/` |
| `--log` | `False` | Export per-frame CSV log to `outputs/` |
| `--no-display` | `False` | Run headless (no GUI window) |

---

## 🔧 Configuration

All defaults live in `config/settings.py`:

```python
# Detection thresholds
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# Target class (COCO: 0 = person)
TARGET_CLASS_ID = 0

# Visualization colors (BGR)
BBOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
HUD_BG_COLOR = (40, 40, 40)
```

---

## 📊 Results

The system outputs:

- **Annotated frames** with bounding boxes and confidence labels
- **Head-count overlay** displayed in real time
- **FPS counter** for performance monitoring
- **CSV log** (`outputs/detection_log.csv`) with columns:

| frame_id | timestamp | person_count | avg_confidence | detections |
|----------|-----------|-------------|----------------|------------|
| 1 | 0.033 | 12 | 0.72 | [bbox_list] |

---

## 🔬 Technical Details

### Detection Pipeline

1. **Frame Capture** — OpenCV `VideoCapture` reads from webcam or file
2. **Preprocessing** — YOLOv8 internally resizes to `640×640` and normalizes
3. **Inference** — Single-shot detection via Ultralytics `model.predict()`
4. **Filtering** — Keep only `class 0` (person) detections above confidence threshold
5. **NMS** — Non-Maximum Suppression at configurable IoU to remove overlaps
6. **Visualization** — Draw bounding boxes, labels, and HUD via OpenCV primitives
7. **Logging** — Append per-frame stats to CSV buffer

### Model Variants

| Model | Size (MB) | mAP@50 | Speed (ms) | Recommended For |
|-------|-----------|--------|------------|-----------------|
| YOLOv8n | 6.2 | 37.3 | 1.2 | Real-time / edge |
| YOLOv8s | 21.5 | 44.9 | 2.1 | Balanced |
| YOLOv8m | 49.7 | 50.2 | 5.0 | Accuracy-first |
| YOLOv8l | 83.7 | 52.9 | 8.5 | High accuracy |
| YOLOv8x | 130.5 | 53.9 | 14.0 | Maximum accuracy |

### Key Dependencies

- **[Ultralytics](https://docs.ultralytics.com/)** — YOLOv8 training & inference SDK
- **[OpenCV](https://opencv.org/)** — Video I/O, image processing, GUI rendering
- **[NumPy](https://numpy.org/)** — Array operations for detection post-processing

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Built with ❤️ using YOLOv8 + OpenCV</b>
</p>
