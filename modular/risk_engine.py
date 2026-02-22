"""
Main risk detection engine combining all detectors
"""
import cv2
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .config import (
    MAX_RISK_SCORE, RISK_RUNNING, RISK_FORKLIFT_PROXIMITY,
    RISK_BAD_LIFT, RISK_SLIP, RISK_NO_HELMET, RISK_NO_VEST,
    DEFAULT_CONFIDENCE,
    COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_YELLOW, COLOR_BLACK, COLOR_GREEN
)
from .detectors.ppe_detector import PPEDetector
from .detectors.pose_detector import PoseAnalyzer
from .detectors.motion_detector import MotionDetector


@dataclass
class FrameResult:
    """Results from processing a single frame."""
    frame_id: int
    risk_score: int
    timestamp: float
    persons_detected: int
    violations: List[str] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None


@dataclass
class VideoResult:
    """Results from processing an entire video."""
    total_frames: int
    final_risk_score: int
    risk_log: List[int] = field(default_factory=list)
    time_log: List[float] = field(default_factory=list)
    violations_summary: Dict[str, int] = field(default_factory=dict)


class RiskEngine:
    """
    Main engine for warehouse/construction safety risk detection.
    
    Combines multiple detectors:
    - PPE violation detection
    - Running detection
    - Forklift proximity detection
    - Bad lifting posture detection
    - Slip risk detection
    """
    
    def __init__(self, base_model, ppe_model, pose_model=None, 
                 confidence: float = DEFAULT_CONFIDENCE):
        """
        Initialize the risk detection engine.
        
        Args:
            base_model: YOLO model for general detection
            ppe_model: YOLO model for PPE detection
            pose_model: YOLO pose model (optional)
            confidence: Detection confidence threshold
        """
        self.confidence = confidence
        
        # Initialize detectors
        self.ppe_detector = PPEDetector(ppe_model, confidence)
        self.motion_detector = MotionDetector(base_model)
        self.pose_analyzer = PoseAnalyzer(pose_model) if pose_model else None
        
        # State
        self.risk_score = 0
        self.heatmap: Optional[np.ndarray] = None
        
        # Violation counts
        self.violation_counts: Dict[str, int] = defaultdict(int)
    
    def reset(self, frame_shape: Tuple[int, int, int] = None) -> None:
        """
        Reset engine state for new video processing.
        
        Args:
            frame_shape: Shape of frames (height, width, channels)
        """
        self.risk_score = 0
        self.violation_counts.clear()
        
        if frame_shape:
            self.heatmap = np.zeros(
                (frame_shape[0], frame_shape[1]), 
                dtype=np.float32
            )
        else:
            self.heatmap = np.zeros((480, 640), dtype=np.float32)
        
        # Reset detector tracking
        self.motion_detector.reset_tracking()
        if self.pose_analyzer:
            self.pose_analyzer.reset_tracking()
    
    def _update_risk(self, value: int) -> None:
        """Update risk score with clamping."""
        self.risk_score = min(MAX_RISK_SCORE, self.risk_score + value)
    
    def _update_heatmap(self, x: int, y: int, weight: int = 1) -> None:
        """Update heatmap at position."""
        if self.heatmap is not None:
            if 0 <= y < self.heatmap.shape[0] and 0 <= x < self.heatmap.shape[1]:
                self.heatmap[y, x] += weight
    
    def _draw_heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Apply heatmap overlay to frame."""
        if self.heatmap is None:
            return frame
        
        hm = cv2.GaussianBlur(self.heatmap, (31, 31), 0)
        hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
        hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.7, hm_color, 0.3, 0)
    
    def process_frame(self, frame: np.ndarray, 
                      frame_id: int = 0,
                      draw_annotations: bool = True) -> FrameResult:
        """
        Process a single frame for all risks.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame number
            draw_annotations: Whether to draw annotations on frame
            
        Returns:
            FrameResult with detections and risk score
        """
        violations = []
        annotated = frame.copy() if draw_annotations else None
        
        # Motion detection FIRST (to get base model person detections)
        motion_risks, base_persons, forklifts = self.motion_detector.detect(
            frame, self.confidence
        )
        
        # Convert base_persons to simple box format for PPE fallback
        base_person_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, cx, cy in base_persons]
        
        # PPE Detection (helmet, vest violations) - with fallback to base model persons
        ppe_result = self.ppe_detector.detect(frame, external_persons=base_person_boxes)
        
        for violation in ppe_result.violations:
            if violation.violation_type == "NO_HELMET":
                self._update_risk(RISK_NO_HELMET)
                self.violation_counts["NO_HELMET"] += 1
                violations.append("NO_HELMET")
                
                if draw_annotations:
                    x1, y1, x2, y2 = violation.person_box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_RED, 2)
                    cv2.putText(annotated, "NO HELMET", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
                    self._update_heatmap((x1 + x2) // 2, (y1 + y2) // 2, 4)
            
            elif violation.violation_type == "NO_VEST":
                self._update_risk(RISK_NO_VEST)
                self.violation_counts["NO_VEST"] += 1
                violations.append("NO_VEST")
                
                if draw_annotations:
                    x1, y1, x2, y2 = violation.person_box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_ORANGE, 2)
                    cv2.putText(annotated, "NO VEST", (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 2)
                    self._update_heatmap((x1 + x2) // 2, (y1 + y2) // 2, 3)
        
        # Draw properly equipped persons in green
        if draw_annotations:
            for person_box in ppe_result.persons:
                x1, y1, x2, y2 = person_box
                # Check if this person has violations
                has_violation = any(
                    v.person_box == person_box for v in ppe_result.violations
                )
                if not has_violation:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_GREEN, 2)
                    cv2.putText(annotated, "PPE OK", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2)
        
        # Process motion risks (running, forklift proximity - already detected above)
        for risk in motion_risks:
            if risk.risk_type == "RUNNING":
                self._update_risk(RISK_RUNNING)
                self.violation_counts["RUNNING"] += 1
                violations.append("RUNNING")
                
                if draw_annotations:
                    x1, y1, x2, y2 = risk.person_box
                    cv2.putText(annotated, "RUNNING", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
            
            elif risk.risk_type == "FORKLIFT_PROXIMITY":
                self._update_risk(RISK_FORKLIFT_PROXIMITY)
                self.violation_counts["FORKLIFT_RISK"] += 1
                violations.append("FORKLIFT_PROXIMITY")
                
                if draw_annotations and risk.related_point:
                    x1, y1, x2, y2 = risk.person_box
                    fx, fy = risk.related_point
                    cv2.line(annotated, risk.person_center, (fx, fy), COLOR_RED, 2)
                    cv2.putText(annotated, "FORKLIFT RISK", (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
        
        # Draw persons and forklifts from base model
        if draw_annotations:
            for x1, y1, x2, y2, cx, cy in base_persons:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_BLUE, 2)
            
            for fx, fy in forklifts:
                cv2.circle(annotated, (fx, fy), 6, COLOR_ORANGE, -1)
                cv2.putText(annotated, "FORKLIFT", (fx - 40, fy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ORANGE, 2)
        
        # Pose analysis
        if self.pose_analyzer:
            pose_risks = self.pose_analyzer.analyze_frame(frame, self.confidence)
            
            for risk in pose_risks:
                if risk.risk_type == "BAD_LIFT":
                    self._update_risk(RISK_BAD_LIFT)
                    self._update_heatmap(risk.location[0], risk.location[1], 3)
                    self.violation_counts["BAD_LIFT"] += 1
                    violations.append("BAD_LIFT")
                    
                    if draw_annotations:
                        cv2.putText(annotated, "BAD LIFT",
                                   (risk.location[0], risk.location[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
                
                elif risk.risk_type == "SLIP_RISK":
                    self._update_risk(RISK_SLIP)
                    self._update_heatmap(risk.location[0], risk.location[1], 2)
                    self.violation_counts["SLIP_RISK"] += 1
                    violations.append("SLIP_RISK")
                    
                    if draw_annotations:
                        cv2.putText(annotated, "SLIP RISK", risk.location,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 2)
        
        # Apply heatmap overlay
        if draw_annotations:
            annotated = self._draw_heatmap_overlay(annotated)
            
            # Draw risk score
            cv2.rectangle(annotated, (10, 10), (300, 70), COLOR_BLACK, -1)
            cv2.putText(annotated, f"INJURY RISK SCORE: {self.risk_score}",
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)
        
        # Use max of PPE and motion detector person counts
        total_persons = max(len(ppe_result.persons), len(base_persons))
        
        return FrameResult(
            frame_id=frame_id,
            risk_score=self.risk_score,
            timestamp=time.time(),
            persons_detected=total_persons,
            violations=violations,
            annotated_frame=annotated
        )
    
    def process_video(self, video_path: str,
                      callback: Callable[[FrameResult], bool] = None,
                      max_frames: int = None) -> VideoResult:
        """
        Process an entire video file.
        
        Args:
            video_path: Path to video file
            callback: Optional callback function called for each frame.
                     Should return True to continue, False to stop.
            max_frames: Maximum frames to process (None for all)
            
        Returns:
            VideoResult with processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get first frame for initialization
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.reset(first_frame.shape)
        
        risk_log = []
        time_log = []
        frame_id = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = self.process_frame(frame, frame_id)
                risk_log.append(result.risk_score)
                time_log.append(result.timestamp)
                
                if callback:
                    if not callback(result):
                        break
                
                frame_id += 1
                
                if max_frames and frame_id >= max_frames:
                    break
        
        finally:
            cap.release()
        
        return VideoResult(
            total_frames=frame_id,
            final_risk_score=self.risk_score,
            risk_log=risk_log,
            time_log=time_log,
            violations_summary=dict(self.violation_counts)
        )
