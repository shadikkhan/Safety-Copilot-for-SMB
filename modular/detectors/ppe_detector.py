"""
PPE (Personal Protective Equipment) violation detection
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image

from ..config import (
    PPE_HELMET_CLASS, PPE_VEST_CLASS, PPE_PERSON_CLASS,
    PPE_NO_HELMET_CLASS, PPE_NO_GOGGLE_CLASS, PPE_NO_GLOVES_CLASS, PPE_NO_BOOTS_CLASS,
    COLOR_RED, COLOR_GREEN, PPE_CONFIDENCE
)


@dataclass
class PPEViolation:
    """Represents a PPE violation detection."""
    violation_type: str  # 'NO_HELMET' or 'NO_VEST'
    person_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float = 0.0


@dataclass
class PPEDetectionResult:
    """Results from PPE detection on an image/frame."""
    persons: List[Tuple[int, int, int, int]]
    helmets: List[Tuple[int, int, int, int]]
    vests: List[Tuple[int, int, int, int]]
    violations: List[PPEViolation]


class PPEDetector:
    """Detector for PPE equipment and violations."""
    
    def __init__(self, model, confidence: float = PPE_CONFIDENCE):
        """
        Initialize PPE detector.
        
        Args:
            model: YOLO model for PPE detection
            confidence: Detection confidence threshold
        """
        self.model = model
        self.confidence = confidence
    
    @staticmethod
    def is_inside(ppe_box: Tuple[int, int, int, int], 
                  person_box: Tuple[int, int, int, int]) -> bool:
        """
        Check if PPE bounding box center is inside person bounding box.
        
        Args:
            ppe_box: (x1, y1, x2, y2) of PPE item
            person_box: (x1, y1, x2, y2) of person
            
        Returns:
            True if PPE center is inside person box
        """
        px1, py1, px2, py2 = person_box
        x1, y1, x2, y2 = ppe_box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return px1 <= cx <= px2 and py1 <= cy <= py2
    
    @staticmethod
    def _boxes_overlap(box1: Tuple[int, int, int, int], 
                       box2: Tuple[int, int, int, int],
                       threshold: float = 0.3) -> bool:
        """
        Check if two bounding boxes overlap significantly.
        
        Args:
            box1: (x1, y1, x2, y2) first box
            box2: (x1, y1, x2, y2) second box
            threshold: IoU threshold for overlap
            
        Returns:
            True if boxes overlap more than threshold
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return (intersection / union) > threshold if union > 0 else False
    
    def detect(self, frame: np.ndarray, 
               external_persons: List[Tuple[int, int, int, int]] = None) -> PPEDetectionResult:
        """
        Detect PPE equipment and violations in a frame.
        
        Args:
            frame: Image frame (BGR format)
            external_persons: Optional list of person boxes from external detector
                             (fallback when PPE model doesn't detect persons)
            
        Returns:
            PPEDetectionResult with persons, PPE items, and violations
        """
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        
        persons = []
        helmets = []
        vests = []
        violations = []
        
        # Direct violation detections from model
        direct_no_helmet_boxes = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            if cls == PPE_PERSON_CLASS:
                persons.append((x1, y1, x2, y2))
            elif cls == PPE_HELMET_CLASS:
                helmets.append((x1, y1, x2, y2))
            elif cls == PPE_VEST_CLASS:
                vests.append((x1, y1, x2, y2))
            elif cls == PPE_NO_HELMET_CLASS:
                # Direct no_helmet detection - use as person box
                direct_no_helmet_boxes.append((x1, y1, x2, y2))
                violations.append(PPEViolation("NO_HELMET", (x1, y1, x2, y2), conf))
        
        # Use external persons as fallback if PPE model didn't detect any
        if not persons and external_persons:
            persons = list(external_persons)
        
        # For persons detected, check if they have helmet/vest
        for person in persons:
            helmet_found = any(self.is_inside(h, person) for h in helmets)
            vest_found = any(self.is_inside(v, person) for v in vests)
            
            # Check if already flagged via direct detection
            already_flagged_no_helmet = any(
                self._boxes_overlap(person, nhb) for nhb in direct_no_helmet_boxes
            )
            
            if not helmet_found and not already_flagged_no_helmet:
                violations.append(PPEViolation("NO_HELMET", person))
            if not vest_found:
                violations.append(PPEViolation("NO_VEST", person))
        
        # Add direct no_helmet boxes to persons list for visualization
        for box in direct_no_helmet_boxes:
            if box not in persons:
                persons.append(box)
        
        return PPEDetectionResult(
            persons=persons,
            helmets=helmets,
            vests=vests,
            violations=violations
        )
    
    def draw_violations(self, frame: np.ndarray, 
                        result: PPEDetectionResult) -> np.ndarray:
        """
        Draw violation annotations on frame.
        
        Args:
            frame: Image frame to annotate
            result: PPE detection result
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for violation in result.violations:
            x1, y1, x2, y2 = violation.person_box
            
            # Draw red rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_RED, 2)
            
            # Draw label
            cv2.putText(
                annotated,
                violation.violation_type,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_RED,
                2
            )
        
        return annotated


def detect_ppe_violations(image_path: str, model, conf: float = 0.4) -> List[PPEViolation]:
    """
    Detect PPE violations in an image file.
    
    Args:
        image_path: Path to the image file
        model: YOLO model for PPE detection
        conf: Confidence threshold
        
    Returns:
        List of detected violations
    """
    detector = PPEDetector(model, conf)
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    result = detector.detect(img)
    return result.violations
