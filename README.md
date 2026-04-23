# TacticEYE2 - Football Match Analysis System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-11-green)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

TacticEYE2 is a real-time football video analytics system with a full web interface and core modules for player tracking, team classification, ball possession, and pass counting.

## What's New

- Complete FastAPI + WebSocket web app for live analysis
- Real-time dashboard with interactive charts (Chart.js + Bootstrap)
- Deterministic possession engine (`PossessionTrackerV2`)
- Automatic pass counter by team
- Improved team classification stability with `TeamClassifierV2`

## Core Features

### 1) Re-Identification Tracking (ReID)
- Player ReID using deep features (OSNet)
- Persistent player IDs across the full match
- Matching based on visual similarity + IoU
- Occlusion and re-entry handling

### 2) Automatic Team Classification
- `TeamClassifierV2`: K-means in LAB color space with green filtering
- Automatic referee detection
- Temporal voting for stable team assignments

### 3) Ball Possession Detection (V2)
- Deterministic possession assignment
- 3-step process: detect ball -> nearest player -> distance validation
- Configurable hysteresis (default: 5 frames)
- Configurable possession distance (default: 60 px)
- Full possession timeline and live stats
- On-frame possession visualization (highlight + ball link)

### 4) Pass Counter
- Automatic pass detection between teammates
- Cumulative team pass statistics
- Live on-screen pass updates

### 5) Web Application
- End-to-end FastAPI web app
- Video upload and live analysis processing
- Real-time updates through WebSocket
- Responsive dashboard with interactive charts

## Requirements

### Hardware
- NVIDIA GPU with CUDA recommended (minimum 6 GB VRAM)
- 8 GB RAM minimum
- 2 GB free storage

### Software
- Python 3.8+
- CUDA 11.8+ (GPU mode)

## Installation

### 1. Clone repository
```bash
git clone https://github.com/TuUsuario/TacticEYE2.git
cd TacticEYE2
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_web.txt
```

### 4. Validate setup
```bash
python setup_check.py
```

## Usage

### Web App (Recommended)

1. Start server:
```bash
python app.py
```

Default port is `8001` (to avoid common `8000` conflicts). If busy, a free port is selected automatically.

2. Open in browser:
```
http://localhost:8001
```

To force a specific port:
```bash
PORT=8010 python app.py
```

3. In the UI:
- Upload a video (`.mp4`, `.avi`, etc.)
- Analysis starts automatically
- Monitor live stats:
  - Team possession (%)
  - Completed passes
  - Possession timeline
  - Interactive charts

### Command Line

Basic run:
```bash
python pruebatrackequipo.py video.mp4 --model weights/best.pt --reid
```

High-precision possession setup:
```bash
python pruebatrackequipo.py video.mp4 \
    --model weights/best.pt \
    --reid \
    --possession-distance 40
```

Faster run without preview window:
```bash
python pruebatrackequipo.py video.mp4 \
    --model weights/best.pt \
    --reid \
    --no-show \
    --output result.mp4
```

## Main CLI Parameters

### YOLO Detection
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | YOLO model path | `yolov8n.pt` |
| `--imgsz` | Input image size | `640` |
| `--conf` | Confidence threshold | `0.35` |
| `--max-det` | Max detections | `100` |

### ReID Tracking
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--reid` | Enable ReID tracker | `False` |

### Team Classification (V2)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tc-kmeans-min-tracks` | Minimum tracks for KMeans | `12` |
| `--tc-vote-history` | Voting history size | `4` |
| `--tc-use-L` | Use L* channel | `True` |
| `--tc-L-weight` | L* channel weight | `0.5` |

### Team Classification (V3, Experimental)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use-v3` | Enable TeamClassifierV3 | `False` |
| `--v3-recalibrate` | Recalibrate every N frames | `300` |
| `--v3-variance` | Use variance features | `True` |
| `--v3-adaptive-thresh` | Adaptive thresholds | `True` |
| `--v3-hysteresis` | Temporal hysteresis | `True` |

### Possession
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--possession-distance` | Max distance (pixels) | `60` |

### Output
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--no-show` | Disable preview window | `False` |
| `--output` | Save processed video | `None` |

## Project Structure

```text
TacticEYE2/
├── modules/
│   ├── reid_tracker.py
│   ├── team_classifier.py
│   ├── team_classifier_v2.py
│   ├── possession_tracker.py
│   └── possession_tracker_v2.py
├── weights/
│   └── best.pt
├── pruebatrackequipo.py
├── app.py
├── setup_check.py
├── config.yaml
├── requirements.txt
└── requirements_web.txt
```

## Troubleshooting

### `KeyError: -1`
Invalid `team_id` values (typically referees) are filtered automatically by the pipeline.

### Inaccurate team classification
Try V3 mode:
```bash
--use-v3 --v3-recalibrate 300
```

### Possession switches too quickly
Reduce possession distance:
```bash
--possession-distance 40
```

### Slow processing
Lower resolution and disable preview:
```bash
--imgsz 416 --no-show
```

## YOLO Classes

- `0`: `player`
- `1`: `ball`
- `2`: `referee`
- `3`: `goalkeeper`

## Roadmap

- Field calibration (2D -> 3D homography)
- Team heatmaps
- Advanced physical stats (distance, speed)
- Professional broadcast overlays
- Full exports (CSV, JSON, NPZ)
- Event detection extensions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m "Add AmazingFeature"`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid)
- Computer Vision community
