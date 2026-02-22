"""
Configuration constants for Safety Copilot
"""
import os

# =====================================
# MODEL PATHS
# =====================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BASE_MODEL_NAME = "yolov8n"
POSE_MODEL_NAME = "yolov8n-pose"
PPE_MODEL_PATH = os.path.join(MODELS_DIR, "ppe_yolov8n_best.pt")

# =====================================
# DETECTION THRESHOLDS
# =====================================
DEFAULT_CONFIDENCE = 0.4
PPE_CONFIDENCE = 0.5

# =====================================
# RUNNING DETECTION
# =====================================
RUNNING_SPEED_THRESHOLD = 50  # pixels/frame
RUNNING_MIN_FRAMES = 10       # minimum frames for sustained motion detection
TRACK_HISTORY_LENGTH = 8      # frames to keep in track history

# =====================================
# FORKLIFT PROXIMITY
# =====================================
FORKLIFT_DISTANCE_THRESHOLD = 500  # pixels

# =====================================
# POSE ANALYSIS
# =====================================
BAD_LIFT_KNEE_ANGLE_THRESHOLD = 150  # degrees (straight leg = bad lift)
FOOT_INSTABILITY_THRESHOLD = 25      # pixels displacement
FOOT_TRACK_LENGTH = 6                # frames for foot tracking

# =====================================
# RISK SCORING
# =====================================
MAX_RISK_SCORE = 1000
RISK_RUNNING = 2
RISK_FORKLIFT_PROXIMITY = 3
RISK_BAD_LIFT = 3
RISK_SLIP = 2
RISK_NO_HELMET = 4
RISK_NO_VEST = 3

# =====================================
# VISUALIZATION
# =====================================
HEATMAP_BLUR_SIZE = 31
HEATMAP_ALPHA = 0.3

# =====================================
# COLORS (BGR format for OpenCV)
# =====================================
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# =====================================
# CLASS IDS (COCO dataset)
# =====================================
COCO_PERSON_CLASS = 0
COCO_CAR_CLASS = 2
COCO_MOTORCYCLE_CLASS = 3  # Often used for forklift detection

# PPE Model Classes
PPE_HELMET_CLASS = 0
PPE_GLOVES_CLASS = 1
PPE_VEST_CLASS = 2
PPE_BOOTS_CLASS = 3
PPE_GOGGLES_CLASS = 4
PPE_NONE_CLASS = 5
PPE_PERSON_CLASS = 6

# Direct PPE Violation Classes (detected by model)
PPE_NO_HELMET_CLASS = 7
PPE_NO_GOGGLE_CLASS = 8
PPE_NO_GLOVES_CLASS = 9
PPE_NO_BOOTS_CLASS = 10

# =====================================
# OLLAMA LLM CONFIGURATION
# =====================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 120  # seconds
