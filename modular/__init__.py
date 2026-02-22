"""
Safety Copilot for SMB - Modular Package
"""
from .config import *
from .models import ModelManager, load_all_models, get_device
from .risk_engine import RiskEngine, FrameResult, VideoResult
from .visualization import plot_risk_over_time, plot_violations_summary
from .detectors import PPEDetector, PoseAnalyzer, MotionDetector

__version__ = "1.0.0"
__all__ = [
    'ModelManager',
    'load_all_models',
    'get_device',
    'RiskEngine',
    'FrameResult',
    'VideoResult',
    'PPEDetector',
    'PoseAnalyzer',
    'MotionDetector',
    'plot_risk_over_time',
    'plot_violations_summary',
]
