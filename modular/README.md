# 🦺 Safety Copilot for SMB

**AI-powered workplace safety monitoring system for construction sites, warehouses, and industrial facilities.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

---

## 📋 Overview

Safety Copilot is an intelligent safety monitoring solution designed for small and medium businesses (SMBs) in construction, warehousing, and industrial sectors. Using state-of-the-art computer vision and AI, it provides real-time detection of safety violations, risk assessment, and actionable recommendations.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| 🪖 **PPE Detection** | Detects missing helmets, vests, goggles, gloves, and boots |
| 🏃 **Running Detection** | Identifies unsafe running behavior in work zones |
| 🚜 **Forklift Proximity** | Alerts when workers are dangerously close to forklifts |
| 🏋️ **Ergonomic Analysis** | Detects bad lifting posture using pose estimation |
| ⚠️ **Slip Risk** | Identifies potential slip hazards through motion analysis |
| 📊 **Risk Scoring** | Real-time cumulative risk tracking with heatmap visualization |
| 🤖 **AI Reports** | LLM-powered safety reports with business recommendations |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for live camera mode)
- [Ollama](https://ollama.ai/) (optional, for AI report generation)
- CUDA-capable GPU (optional, for faster inference)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd construction

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r modular/requirements.txt
```

### Running the Application

```bash
# From project root
.venv/bin/python -m streamlit run modular/app.py

# Or from modular directory
cd modular
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🖥️ Application Features

### Analysis Modes

| Mode | Description |
|------|-------------|
| **📷 Image Analysis** | Upload and analyze construction/warehouse images for PPE violations |
| **🎬 Video Analysis** | Process recorded videos with frame-by-frame violation tracking |
| **📹 Live Camera** | Real-time monitoring using your webcam or IP camera |
| **📊 Reports** | Aggregated safety analytics and exportable reports |

### AI-Powered Reporting

Safety Copilot integrates with **Ollama** (local LLM) to generate:

- 📋 Executive summaries of safety incidents
- ⚠️ Risk assessments with severity ratings
- 💡 Actionable safety recommendations
- 💰 Business impact analysis
- ✅ Prioritized action items

To enable AI reports:
```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama server
ollama serve

# Pull the required model
ollama pull llama3.1:8b
```

---

## 🧠 Detection Models

| Model | Purpose | Source |
|-------|---------|--------|
| `yolov8n.pt` | Base object detection (persons, vehicles) | Auto-downloaded |
| `yolov8n-pose.pt` | Human pose estimation for ergonomic analysis | Auto-downloaded |
| `ppe_yolov8n_best.pt` | Custom PPE detection model | Trained on construction-ppe dataset |

### PPE Model Classes

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | helmet | 6 | Person |
| 1 | gloves | 7 | **no_helmet** ⚠️ |
| 2 | vest | 8 | no_goggle |
| 3 | boots | 9 | no_gloves |
| 4 | goggles | 10 | no_boots |
| 5 | none | | |

---

## 📁 Project Structure

```
construction/
├── modular/                    # Main application package
│   ├── app.py                  # Streamlit web application
│   ├── config.py               # Configuration constants
│   ├── models.py               # Model loading utilities
│   ├── risk_engine.py          # Main risk detection engine
│   ├── llm_reporter.py         # Ollama LLM integration
│   ├── visualization.py        # Plotting and visualization
│   ├── requirements.txt        # Python dependencies
│   └── detectors/
│       ├── ppe_detector.py     # PPE violation detection
│       ├── pose_detector.py    # Posture/ergonomic analysis
│       └── motion_detector.py  # Running & proximity detection
│
├── models/                     # Model files directory
│   ├── ppe_yolov8n_best.pt     # Custom trained PPE model
│   ├── yolov8n.pt              # Base YOLO model
│   └── yolov8n-pose.pt         # Pose estimation model
│
├── construction-ppe/           # PPE training dataset
│   ├── data.yaml               # Dataset configuration
│   ├── images/                 # Training/validation images
│   └── labels/                 # YOLO format annotations
│
├── Safe-and-Unsafe-Behaviours-Dataset/  # Behavior classification dataset
│
├── runs/                       # Training run outputs
├── images/                     # Test images
├── vid_test/                   # Test videos
└── safety_copilot_for_smb.ipynb  # Development notebook
```

---

## ⚙️ Configuration

All thresholds and settings are in `modular/config.py`:

```python
# Detection Confidence
DEFAULT_CONFIDENCE = 0.4
PPE_CONFIDENCE = 0.5

# Running Detection
RUNNING_SPEED_THRESHOLD = 50      # pixels/frame
RUNNING_MIN_FRAMES = 10           # minimum frames for detection

# Forklift Safety
FORKLIFT_DISTANCE_THRESHOLD = 500 # pixels (danger zone)

# Ergonomic Analysis
BAD_LIFT_KNEE_ANGLE_THRESHOLD = 150  # degrees (straight leg = bad lift)

# Risk Scoring Weights
RISK_NO_HELMET = 4   # Critical PPE
RISK_NO_VEST = 3     # High visibility
RISK_FORKLIFT_PROXIMITY = 3
RISK_BAD_LIFT = 3
RISK_RUNNING = 2
RISK_SLIP = 2

# LLM Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 120  # seconds
```

---

## 🐍 Python API

### Full Pipeline

```python
from modular import ModelManager, RiskEngine

# Load all models
manager = ModelManager()
manager.load_all()

# Create risk engine
engine = RiskEngine(
    manager.base_model,
    manager.ppe_model,
    manager.pose_model
)

# Process video
result = engine.process_video("path/to/video.mp4")
print(f"Final Risk Score: {result.final_risk_score}")
print(f"Violations: {result.violations_summary}")
```

### Individual Detectors

```python
from modular.detectors import PPEDetector, PoseAnalyzer, MotionDetector
from modular.models import load_ppe_model, load_pose_model, load_base_model
import cv2

frame = cv2.imread("image.jpg")

# PPE Detection
ppe_model = load_ppe_model()
detector = PPEDetector(ppe_model)
result = detector.detect(frame)
print(f"Violations: {[v.violation_type for v in result.violations]}")

# Pose Analysis
pose_model = load_pose_model()
analyzer = PoseAnalyzer(pose_model)
risks = analyzer.analyze_frame(frame)

# Motion Detection
base_model = load_base_model()
motion = MotionDetector(base_model)
risks, persons, forklifts = motion.detect(frame)
```

### LLM Report Generation

```python
from modular.llm_reporter import LLMReporter

reporter = LLMReporter()

if reporter.is_available():
    report = reporter.generate_report(
        violation_counts={"Missing Helmet": 5, "Missing Vest": 3},
        total_frames=1000,
        duration_seconds=60,
        final_risk_score=75,
        context="construction site"
    )
    print(report.summary)
    print(report.recommendations)
```

---

## 📊 Training Custom Models

The PPE model was trained using the included `construction-ppe` dataset:

```python
from ultralytics import YOLO

# Train PPE detection model
model = YOLO('yolov8n.pt')
results = model.train(
    data='construction-ppe/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

See `safety_copilot_for_smb.ipynb` for the full training workflow.

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Webcam not detected** | Check permissions in System Preferences → Privacy → Camera |
| **CUDA out of memory** | Reduce batch size or use CPU inference |
| **Ollama not responding** | Ensure `ollama serve` is running on port 11434 |
| **PPE not detecting persons** | The system falls back to base YOLO model for person detection |
| **Slow inference** | Install CUDA toolkit for GPU acceleration |

---

## 📦 Dependencies

- **ultralytics** - YOLOv8 models
- **torch/torchvision** - PyTorch deep learning
- **opencv-python** - Computer vision
- **streamlit** - Web UI framework
- **matplotlib/seaborn** - Visualization
- **numpy/scipy** - Numerical computing

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Ollama](https://ollama.ai/) for local LLM inference
- [Streamlit](https://streamlit.io/) for the web framework
- Construction PPE dataset contributors
