"""Session history tracking for language detection."""

from dataclasses import dataclass
from datetime import datetime
from typing import List
from pathlib import Path


@dataclass
class DetectionResult:
    language: str
    probability: float
    timestamp: datetime
    audio_duration: float
    source: str
    
    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] {self.language.capitalize()} - {int(self.probability * 100)}%"


class SessionHistory:
    """Manages language detection history."""
    
    def __init__(self):
        self.detections: List[DetectionResult] = []
    
    def add_detection(self, result: DetectionResult) -> None:
        self.detections.append(result)
    
    def get_recent(self, n: int = 3) -> List[DetectionResult]:
        return self.detections[-n:] if self.detections else []
    
    def clear(self) -> None:
        self.detections = []
    
    def export_to_file(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Language Detection History\n")
            f.write("=" * 60 + "\n\n")
            
            if not self.detections:
                f.write("No detections.\n")
                return
            
            f.write(f"Total: {len(self.detections)}\n")
            f.write(f"Start: {self.detections[0].timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End: {self.detections[-1].timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, d in enumerate(self.detections, 1):
                f.write(f"#{i} {d.timestamp.strftime('%H:%M:%S')} - "
                       f"{d.language.capitalize()} ({d.probability*100:.0f}%) - "
                       f"{d.source}\n")
