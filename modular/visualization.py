"""
Visualization utilities for Safety Copilot
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image

from .config import COLOR_RED, COLOR_GREEN


def plot_risk_over_time(risk_log: List[int], 
                        title: str = "Injury Risk Over Time",
                        figsize: Tuple[int, int] = (10, 4),
                        save_path: str = None) -> plt.Figure:
    """
    Plot risk score over time.
    
    Args:
        risk_log: List of risk scores per frame
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(risk_log, color='red', linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Risk Score')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(risk_log) * 1.1 if risk_log else 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_violations_summary(violations: Dict[str, int],
                           title: str = "Violations Summary",
                           figsize: Tuple[int, int] = (8, 6),
                           save_path: str = None) -> plt.Figure:
    """
    Plot bar chart of violation counts.
    
    Args:
        violations: Dictionary of violation type -> count
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if not violations:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No violations detected', 
                ha='center', va='center', fontsize=14)
        ax.set_title(title)
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = list(violations.keys())
    values = list(violations.values())
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(labels)))
    
    bars = ax.bar(labels, values, color=colors, edgecolor='darkred')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Violation Type')
    ax.set_ylabel('Count')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_ppe_detection(image_path: str,
                           violations: List[Any],
                           persons_count: int,
                           helmets_count: int,
                           vests_count: int,
                           figsize: Tuple[int, int] = (20, 8)) -> plt.Figure:
    """
    Visualize PPE detection results with side-by-side comparison.
    
    Args:
        image_path: Path to image
        violations: List of violation objects
        persons_count, helmets_count, vests_count: Detection counts
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    img = Image.open(image_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Original image
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    
    # Image with violations
    ax2.imshow(img)
    
    import matplotlib.patches as patches
    
    for violation in violations:
        x1, y1, x2, y2 = violation.person_box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor='red', facecolor='none'
        )
        ax2.add_patch(rect)
        ax2.text(x1, y1 - 10, violation.violation_type,
                color='red', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', 
                         facecolor='white', alpha=0.8))
    
    ax2.axis('off')
    
    if violations:
        ax2.set_title('PPE Violations Detected', 
                     fontsize=14, fontweight='bold', color='red')
    else:
        ax2.set_title('No Violations Found', 
                     fontsize=14, fontweight='bold', color='green')
    
    plt.tight_layout()
    
    return fig


def create_video_frame_with_stats(frame: np.ndarray,
                                  risk_score: int,
                                  frame_id: int,
                                  persons_count: int,
                                  violations: List[str]) -> np.ndarray:
    """
    Add statistics overlay to video frame.
    
    Args:
        frame: Video frame
        risk_score: Current risk score
        frame_id: Frame number
        persons_count: Number of persons detected
        violations: List of current violations
        
    Returns:
        Frame with overlay
    """
    result = frame.copy()
    height, width = result.shape[:2]
    
    # Stats panel
    panel_height = 100
    cv2.rectangle(result, (0, 0), (width, panel_height), (0, 0, 0), -1)
    cv2.rectangle(result, (0, 0), (width, panel_height), (255, 255, 255), 2)
    
    # Risk score with color coding
    risk_color = (0, 255, 0) if risk_score < 30 else \
                 (0, 255, 255) if risk_score < 60 else \
                 (0, 0, 255)
    
    cv2.putText(result, f"RISK: {risk_score}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1, risk_color, 2)
    
    cv2.putText(result, f"Frame: {frame_id} | Persons: {persons_count}",
               (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Violations on right side
    if violations:
        x_offset = width - 250
        cv2.putText(result, "ALERTS:", (x_offset, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for i, v in enumerate(violations[:3]):  # Show max 3
            cv2.putText(result, f"- {v}", (x_offset, 55 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return result


def frame_to_bytes(frame: np.ndarray, format: str = 'jpeg') -> bytes:
    """
    Convert OpenCV frame to bytes for streaming.
    
    Args:
        frame: BGR frame
        format: Image format ('jpeg' or 'png')
        
    Returns:
        Encoded image bytes
    """
    if format == 'jpeg':
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    else:
        _, buffer = cv2.imencode('.png', frame)
    return buffer.tobytes()


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
