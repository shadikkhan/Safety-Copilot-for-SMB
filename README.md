# 🦺 Safety Copilot for SMB

**AI-Powered Workplace Safety Monitoring System for Construction Sites and Industrial Facilities**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com/)

---

## 📋 Overview

Safety Copilot is an intelligent safety monitoring solution designed for small and medium businesses (SMBs) in construction, warehousing, and industrial sectors. Using state-of-the-art computer vision and AI, it provides real-time detection of safety violations, risk assessment, and actionable recommendations.

### ✨ Key Features

- **🪖 PPE Detection**: Real-time detection of missing helmets, vests, goggles, gloves, and boots
- **🏃 Running Detection**: Identifies unsafe running behavior in work zones
- **🚜 Forklift Proximity Alerts**: Detects workers dangerously close to forklifts
- **🏋️ Ergonomic Analysis**: Bad lifting posture detection using pose estimation
- **⚠️ Slip Risk Detection**: Identifies potential slip hazards through motion analysis
- **📊 Real-time Risk Scoring**: Cumulative risk tracking with heatmap visualization
- **🤖 AI-Powered Reports**: LLM-generated safety reports with business recommendations
- **📹 Multiple Input Modes**: Image, video, and live camera analysis

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Webcam** (for live camera mode)
- **[Ollama](https://ollama.ai/)** (optional, for AI report generation)
- CUDA-capable GPU (optional, for faster inference)

### Installation

```bash
# Clone the repository
git clone <repository-url>


# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r modular/requirements.txt
```

### Running the Application

```bash
# From project root
streamlit run modular/app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🖥️ Usage

### Analysis Modes

| Mode | Description |
|------|-------------|
| **📷 Image Analysis** | Upload construction/warehouse images for PPE violation detection |
| **🎬 Video Analysis** | Process recorded videos with frame-by-frame violation tracking |
| **📹 Live Camera** | Real-time monitoring using webcam or IP camera streams |
| **📊 Reports** | Aggregated safety analytics and exportable reports |

### Enabling AI Reports (Optional)

To enable LLM-powered safety reports, install and configure Ollama:

```bash
# Install Ollama
# macOS:
brew install ollama

# Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull a language model
ollama pull llama3.1:8b
```

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
│   ├── README.md               # Detailed module documentation
│   └── detectors/              # Detection modules
│       ├── ppe_detector.py     # PPE violation detection
│       ├── pose_detector.py    # Posture/ergonomic analysis
│       └── motion_detector.py  # Running & proximity detection
│
├── models/                     # Pre-trained model files (not in git)
├── construction-ppe.yaml       # Dataset configuration
├── safety_copilot_for_smb.ipynb  # Jupyter notebook for training/testing
└── .gitignore                  # Git exclusions for datasets and models
```

---

## 🧠 Technology Stack

### Core Technologies

- **Computer Vision**: YOLOv8 (Ultralytics) for object detection and pose estimation
- **Deep Learning**: PyTorch for model inference
- **Web Framework**: Streamlit for interactive UI
- **Image Processing**: OpenCV for video/image manipulation
- **AI Integration**: Ollama for local LLM-powered reporting

### Detection Models

| Model | Purpose | Type |
|-------|---------|------|
| `yolov8n.pt` | Base object detection (persons, vehicles) | Pre-trained |
| `yolov8n-pose.pt` | Human pose estimation for ergonomic analysis | Pre-trained |
| `ppe_yolov8n_best.pt` | Custom PPE detection (helmets, vests, etc.) | Custom-trained |

---

## 🔬 Safety Violations Detected

### PPE Violations
- ❌ Missing helmet
- ❌ Missing safety vest
- ❌ Missing goggles
- ❌ Missing gloves
- ❌ Missing safety boots

### Behavioral Violations
- 🏃 Running in work zone
- 🚜 Too close to forklift
- 🏋️ Bad lifting posture
- 💦 Slip risk detected

### Risk Scoring
Each violation is assigned a severity score:
- **Critical**: 25-30 points (e.g., missing helmet, forklift proximity)
- **High**: 15-20 points (e.g., running, bad posture)
- **Medium**: 10 points (e.g., missing gloves)
- **Low**: 5 points (e.g., missing goggles)

---

## 📊 Output Features

- **Real-time annotations**: Bounding boxes and labels on detected violations
- **Risk heatmap**: Visual representation of cumulative risk over time
- **Violation timeline**: Frame-by-frame violation tracking for videos
- **Summary statistics**: Counts and percentages of each violation type
- **AI-generated reports**: Natural language safety reports with recommendations
- **Export options**: Save annotated videos and analysis results

---

## 🛠️ Development

### Training Custom Models

The PPE detection model was trained on a custom construction-ppe dataset. To retrain or fine-tune:

1. Prepare your dataset in YOLO format
2. Update `construction-ppe.yaml` with dataset paths
3. Use the provided Jupyter notebook: `safety_copilot_for_smb.ipynb`
4. Place trained models in the `models/` directory

### Adding New Detectors

To add a new safety violation detector:

1. Create a new detector class in `modular/detectors/`
2. Implement the detection logic following existing patterns
3. Register the detector in `modular/risk_engine.py`
4. Add corresponding risk scores in `modular/config.py`

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** for the excellent YOLOv8 framework
- **Streamlit** for the intuitive web framework
- **Ollama** for local LLM integration
- Construction-PPE dataset contributors

---

## 📧 Contact

For questions, issues, or feature requests, please open an issue on GitHub.

---

## 🚧 Future Enhancements

- [ ] Multi-camera support for large facilities
- [ ] Cloud deployment options (AWS, Azure, GCP)
- [ ] Mobile app for on-site supervisors
- [ ] Historical trend analysis and reporting
- [ ] Integration with existing safety management systems
- [ ] Multi-language support
- [ ] Automated alert notifications (email, SMS, Slack)
- [ ] Worker identification and tracking
- [ ] Customizable safety rules and thresholds

---

**Built with ❤️ for safer workplaces**
