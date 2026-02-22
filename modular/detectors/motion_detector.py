"""
Motion detection for running and proximity analysis
"""
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

from ..config import (
    RUNNING_SPEED_THRESHOLD,
    RUNNING_MIN_FRAMES,
    TRACK_HISTORY_LENGTH,
    FORKLIFT_DISTANCE_THRESHOLD,
    COCO_PERSON_CLASS,
    COCO_MOTORCYCLE_CLASS
)


@dataclass
class MotionRisk:
    """Represents a motion-related risk detection."""
    risk_type: str  # 'RUNNING' or 'FORKLIFT_PROXIMITY'
    person_box: Tuple[int, int, int, int]
    person_center: Tuple[int, int]
    related_point: Optional[Tuple[int, int]] = None  # e.g., forklift position


class MotionDetector:
    """Detector for motion-based risks (running, proximity to machinery)."""
    
    def __init__(self, model):
        """
        Initialize motion detector.
        
        Args:
            model: YOLO model for object detection
        """
        self.model = model
        self.person_tracks: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=TRACK_HISTORY_LENGTH)
        )
        self.next_person_id = 0
        self.person_id_map: Dict[Tuple[int, int], int] = {}
    
    @staticmethod
    def euclidean(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.dist(p1, p2)
    
    @staticmethod
    def compute_average_speed(track: deque) -> float:
        """
        Compute average speed over track history.
        
        Args:
            track: Deque of (x, y) positions
            
        Returns:
            Average speed in pixels per frame
        """
        if len(track) < 2:
            return 0.0
        
        speeds = [
            math.dist(track[i - 1], track[i])
            for i in range(1, len(track))
        ]
        return sum(speeds) / len(speeds) if speeds else 0.0
    
    def is_running(self, track: deque) -> bool:
        """
        Detect if person is running based on track history.
        
        Improved logic:
        1. Need enough frames (filters new detections)
        2. Average speed must be high (sustained motion, not jitter)
        3. Total displacement must be significant (not jittering in place)
        
        Args:
            track: Deque of (x, y) positions
            
        Returns:
            True if running detected
        """
        if len(track) < RUNNING_MIN_FRAMES:
            return False
        
        avg_speed = self.compute_average_speed(track)
        total_displacement = math.dist(track[0], track[-1])
        min_displacement = RUNNING_SPEED_THRESHOLD * (len(track) - 1) * 0.5
        
        return (avg_speed > RUNNING_SPEED_THRESHOLD and 
                total_displacement > min_displacement)
    
    def _match_person_id(self, center: Tuple[int, int], 
                         threshold: int = 50) -> int:
        """Match center point to existing person ID or create new one."""
        for prev_center, pid in self.person_id_map.items():
            if self.euclidean(center, prev_center) < threshold:
                return pid
        
        # New person
        new_id = self.next_person_id
        self.next_person_id += 1
        return new_id
    
    def detect(self, frame, conf: float = 0.4) -> Tuple[List[MotionRisk], 
                                                        List[Tuple[int, int, int, int, int, int]],
                                                        List[Tuple[int, int]]]:
        """
        Detect motion-related risks in a frame.
        
        Args:
            frame: Image frame (BGR format)
            conf: Detection confidence threshold
            
        Returns:
            Tuple of (risks, persons, forklifts)
            - risks: List of MotionRisk objects
            - persons: List of (x1, y1, x2, y2, cx, cy) for each person
            - forklifts: List of (cx, cy) for each forklift
        """
        results = self.model(frame, conf=conf, verbose=False)[0]
        
        persons = []
        forklifts = []
        risks = []
        new_id_map = {}
        
        # Extract detections
        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if cls == COCO_PERSON_CLASS:
                persons.append((x1, y1, x2, y2, cx, cy))
            elif cls == COCO_MOTORCYCLE_CLASS:  # Used for forklift
                forklifts.append((cx, cy))
        
        # Process each person
        for x1, y1, x2, y2, cx, cy in persons:
            center = (cx, cy)
            pid = self._match_person_id(center)
            new_id_map[center] = pid
            
            # Update track
            self.person_tracks[pid].append(center)
            
            # Check for running
            if self.is_running(self.person_tracks[pid]):
                risks.append(MotionRisk(
                    risk_type="RUNNING",
                    person_box=(x1, y1, x2, y2),
                    person_center=center
                ))
            
            # Check forklift proximity
            for fx, fy in forklifts:
                dist = self.euclidean(center, (fx, fy))
                if dist < FORKLIFT_DISTANCE_THRESHOLD:
                    risks.append(MotionRisk(
                        risk_type="FORKLIFT_PROXIMITY",
                        person_box=(x1, y1, x2, y2),
                        person_center=center,
                        related_point=(fx, fy)
                    ))
        
        # Update ID map
        self.person_id_map = new_id_map
        
        return risks, persons, forklifts
    
    def reset_tracking(self) -> None:
        """Reset all tracking state."""
        self.person_tracks.clear()
        self.person_id_map.clear()
        self.next_person_id = 0
