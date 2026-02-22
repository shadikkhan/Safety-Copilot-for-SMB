"""
Model loading and management utilities
"""
import os
import torch
from ultralytics import YOLO
from typing import Optional, Tuple

from .config import (
    MODELS_DIR, BASE_MODEL_NAME, POSE_MODEL_NAME, PPE_MODEL_PATH
)


def get_device() -> str:
    """Check if GPU is available and return appropriate device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download_model(model_name: str, download_dir: str = MODELS_DIR) -> str:
    """
    Download YOLO model if not already present.
    
    Args:
        model_name: Name of the model (e.g., 'yolov8n')
        download_dir: Directory to save models
        
    Returns:
        Path to the downloaded model file
    """
    file_path = os.path.join(download_dir, f"{model_name}.pt")
    
    # Create directory if not exists
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Created directory: {download_dir}")
    
    # Download if not exists
    if not os.path.exists(file_path):
        print(f"Downloading {model_name}.pt to {download_dir}...")
        model = YOLO(model_name)
        model.save(file_path)
        print(f"Downloaded {model_name}.pt successfully.")
    else:
        print(f"{model_name}.pt already exists. Skipping download.")
    
    return file_path


def load_base_model() -> YOLO:
    """Load the base YOLOv8 model for general detection."""
    model_path = download_model(BASE_MODEL_NAME)
    return YOLO(model_path)


def load_pose_model() -> YOLO:
    """Load the YOLOv8 pose estimation model."""
    model_path = download_model(POSE_MODEL_NAME)
    return YOLO(model_path)


def load_ppe_model() -> YOLO:
    """Load the custom PPE detection model."""
    if not os.path.exists(PPE_MODEL_PATH):
        raise FileNotFoundError(
            f"PPE model not found at {PPE_MODEL_PATH}. "
            "Please train or download the PPE model first."
        )
    return YOLO(PPE_MODEL_PATH)


def load_all_models() -> Tuple[YOLO, YOLO, YOLO]:
    """
    Load all required models.
    
    Returns:
        Tuple of (base_model, ppe_model, pose_model)
    """
    print("Loading models...")
    base_model = load_base_model()
    ppe_model = load_ppe_model()
    pose_model = load_pose_model()
    print("All models loaded successfully!")
    return base_model, ppe_model, pose_model


class ModelManager:
    """Singleton class to manage model instances."""
    
    _instance = None
    _base_model: Optional[YOLO] = None
    _ppe_model: Optional[YOLO] = None
    _pose_model: Optional[YOLO] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def base_model(self) -> YOLO:
        if self._base_model is None:
            self._base_model = load_base_model()
        return self._base_model
    
    @property
    def ppe_model(self) -> YOLO:
        if self._ppe_model is None:
            self._ppe_model = load_ppe_model()
        return self._ppe_model
    
    @property
    def pose_model(self) -> YOLO:
        if self._pose_model is None:
            self._pose_model = load_pose_model()
        return self._pose_model
    
    def load_all(self) -> None:
        """Pre-load all models."""
        _ = self.base_model
        _ = self.ppe_model
        _ = self.pose_model
