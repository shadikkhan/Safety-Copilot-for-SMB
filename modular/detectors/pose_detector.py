"""
Pose analysis for posture and stability detection
"""
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import defaultdict, deque

from ..config import (
    BAD_LIFT_KNEE_ANGLE_THRESHOLD,
    FOOT_INSTABILITY_THRESHOLD,
    FOOT_TRACK_LENGTH
)


@dataclass
class PoseRisk:
    """Represents a pose-related risk detection."""
    risk_type: str  # 'BAD_LIFT' or 'SLIP_RISK'
    location: Tuple[int, int]  # (x, y) pixel location
    person_id: int
    confidence: float = 0.0


class PoseAnalyzer:
    """Analyzer for pose-based risk detection."""
    
    # COCO keypoint indices
    LEFT_HIP = 11
    LEFT_KNEE = 13
    LEFT_ANKLE = 15
    RIGHT_HIP = 12
    RIGHT_KNEE = 14
    RIGHT_ANKLE = 16
    
    def __init__(self, model):
        """
        Initialize pose analyzer.
        
        Args:
            model: YOLO pose model
        """
        self.model = model
        self.foot_tracks: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=FOOT_TRACK_LENGTH)
        )
    
    @staticmethod
    def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate angle at point b given three points a-b-c.
        
        Args:
            a, b, c: Points as numpy arrays
            
        Returns:
            Angle in degrees
        """
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        
        norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm_product < 1e-6:
            return 0.0
        
        cosine = np.dot(ba, bc) / norm_product
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))
    
    def check_bad_lifting_posture(self, kpts: np.ndarray) -> bool:
        """
        Check if person has bad lifting posture (straight legs while lifting).
        
        Args:
            kpts: Pose keypoints (17x2 or 17x3 array)
            
        Returns:
            True if bad lifting posture detected
        """
        try:
            # Check left leg
            hip = kpts[self.LEFT_HIP][:2]
            knee = kpts[self.LEFT_KNEE][:2]
            ankle = kpts[self.LEFT_ANKLE][:2]
            
            # Skip if keypoints are not visible (zeros)
            if np.all(hip == 0) or np.all(knee == 0) or np.all(ankle == 0):
                return False
            
            knee_angle = self.angle(hip, knee, ankle)
            
            # Healthy squat: 70-100°, Straight leg > 150° = bad lift
            return knee_angle > BAD_LIFT_KNEE_ANGLE_THRESHOLD
            
        except (IndexError, ValueError):
            return False
    
    def check_foot_instability(self, person_id: int, ankle: Tuple[int, int]) -> bool:
        """
        Check for foot instability (potential slip risk).
        
        Args:
            person_id: ID for tracking person across frames
            ankle: Current ankle position (x, y)
            
        Returns:
            True if instability detected
        """
        self.foot_tracks[person_id].append(ankle)
        track = self.foot_tracks[person_id]
        
        if len(track) < 3:
            return False
        
        # Calculate displacement between consecutive frames
        displacements = [
            math.dist(track[i], track[i + 1])
            for i in range(len(track) - 1)
        ]
        
        # High max displacement indicates sudden movement / slip
        return max(displacements) > FOOT_INSTABILITY_THRESHOLD
    
    def analyze_frame(self, frame: np.ndarray, 
                      conf: float = 0.4) -> List[PoseRisk]:
        """
        Analyze poses in a frame for risks.
        
        Args:
            frame: Image frame (BGR format)
            conf: Detection confidence threshold
            
        Returns:
            List of detected pose risks
        """
        results = self.model(frame, conf=conf, verbose=False)[0]
        risks = []
        
        if results.keypoints is None:
            return risks
        
        for person_id, kpts in enumerate(results.keypoints.xy):
            kpts_np = kpts.cpu().numpy()
            
            # Check bad lifting posture
            if self.check_bad_lifting_posture(kpts_np):
                hip_pos = kpts_np[self.LEFT_HIP]
                if not np.all(hip_pos == 0):
                    risks.append(PoseRisk(
                        risk_type="BAD_LIFT",
                        location=(int(hip_pos[0]), int(hip_pos[1])),
                        person_id=person_id
                    ))
            
            # Check foot instability
            try:
                ankle = kpts_np[self.LEFT_ANKLE]
                if not np.all(ankle == 0):
                    ankle_pos = (int(ankle[0]), int(ankle[1]))
                    if self.check_foot_instability(person_id, ankle_pos):
                        risks.append(PoseRisk(
                            risk_type="SLIP_RISK",
                            location=ankle_pos,
                            person_id=person_id
                        ))
            except (IndexError, ValueError):
                pass
        
        return risks
    
    def reset_tracking(self) -> None:
        """Reset foot tracking history."""
        self.foot_tracks.clear()
