"""
Session History Module

This module provides functionality for tracking and managing language detection
results during a real-time recognition session.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List
from pathlib import Path


@dataclass
class DetectionResult:
    """
    Represents a single language detection result.
    
    Attributes:
        language: Detected language name (e.g., 'english', 'serbian')
        probability: Confidence score for the detection (0.0 to 1.0)
        timestamp: When the detection occurred
        audio_duration: Duration of the audio segment in seconds
        source: Source of the audio ('microphone' or 'file')
    """
    language: str
    probability: float
    timestamp: datetime
    audio_duration: float
    source: str
    
    def __str__(self) -> str:
        """Format detection result as a readable string."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        prob_percent = int(self.probability * 100)
        return f"[{time_str}] {self.language.capitalize()} - {prob_percent}%"


class SessionHistory:
    """
    Manages the history of language detections during a session.
    
    This class stores detection results and provides methods to retrieve,
    clear, and export the detection history.
    """
    
    def __init__(self):
        """Initialize an empty session history."""
        self.detections: List[DetectionResult] = []
    
    def add_detection(self, result: DetectionResult) -> None:
        """
        Add a detection result to the history.
        
        Args:
            result: DetectionResult object to add to history
        """
        self.detections.append(result)
    
    def get_recent(self, n: int = 3) -> List[DetectionResult]:
        """
        Get the most recent n detections.
        
        Args:
            n: Number of recent detections to return (default: 3)
            
        Returns:
            List of the most recent DetectionResult objects (up to n items)
        """
        return self.detections[-n:] if self.detections else []
    
    def clear(self) -> None:
        """Clear all detections from the history."""
        self.detections = []
    
    def export_to_file(self, filepath: str) -> None:
        """
        Export the detection history to a text file.
        
        Args:
            filepath: Path where the history file should be saved
            
        Raises:
            IOError: If the file cannot be written
        """
        path = Path(filepath)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Language Detection Session History\n")
            f.write("=" * 60 + "\n\n")
            
            if not self.detections:
                f.write("No detections recorded in this session.\n")
                return
            
            f.write(f"Total Detections: {len(self.detections)}\n")
            f.write(f"Session Start: {self.detections[0].timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session End: {self.detections[-1].timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "-" * 60 + "\n\n")
            
            # Write each detection
            for i, detection in enumerate(self.detections, 1):
                f.write(f"Detection #{i}\n")
                f.write(f"  Time: {detection.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  Language: {detection.language.capitalize()}\n")
                f.write(f"  Confidence: {detection.probability * 100:.2f}%\n")
                f.write(f"  Audio Duration: {detection.audio_duration:.2f}s\n")
                f.write(f"  Source: {detection.source.capitalize()}\n")
                f.write("\n")
            
            f.write("-" * 60 + "\n")
            f.write(f"End of Report\n")
