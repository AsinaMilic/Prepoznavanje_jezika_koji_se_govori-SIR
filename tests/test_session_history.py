"""
Unit tests for SessionHistory.

Tests cover:
- Adding detections to history
- get_recent returns correct number of items
- clear removes all detections
- export_to_file creates valid text file
"""

import unittest
from datetime import datetime
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.session_history import SessionHistory, DetectionResult


class TestDetectionResult(unittest.TestCase):
    """Test suite for DetectionResult dataclass"""
    
    def test_detection_result_creation(self):
        """Test creating a DetectionResult instance"""
        timestamp = datetime.now()
        result = DetectionResult(
            language='english',
            probability=0.85,
            timestamp=timestamp,
            audio_duration=3.0,
            source='microphone'
        )
        
        self.assertEqual(result.language, 'english')
        self.assertEqual(result.probability, 0.85)
        self.assertEqual(result.timestamp, timestamp)
        self.assertEqual(result.audio_duration, 3.0)
        self.assertEqual(result.source, 'microphone')
    
    def test_detection_result_str_formatting(self):
        """Test DetectionResult string formatting"""
        timestamp = datetime(2024, 1, 15, 14, 30, 45)
        result = DetectionResult(
            language='english',
            probability=0.85,
            timestamp=timestamp,
            audio_duration=3.0,
            source='microphone'
        )
        
        result_str = str(result)
        self.assertIn('[14:30:45]', result_str)
        self.assertIn('English', result_str)
        self.assertIn('85%', result_str)


class TestSessionHistory(unittest.TestCase):
    """Test suite for SessionHistory class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.history = SessionHistory()
        self.timestamp1 = datetime(2024, 1, 15, 14, 30, 0)
        self.timestamp2 = datetime(2024, 1, 15, 14, 30, 3)
        self.timestamp3 = datetime(2024, 1, 15, 14, 30, 6)
        self.timestamp4 = datetime(2024, 1, 15, 14, 30, 9)
        self.timestamp5 = datetime(2024, 1, 15, 14, 30, 12)
    
    def test_initialization(self):
        """Test SessionHistory initialization"""
        history = SessionHistory()
        self.assertIsInstance(history.detections, list)
        self.assertEqual(len(history.detections), 0)
    
    def test_add_detection(self):
        """Test adding a single detection to history"""
        result = DetectionResult(
            language='english',
            probability=0.85,
            timestamp=self.timestamp1,
            audio_duration=3.0,
            source='microphone'
        )
        
        self.history.add_detection(result)
        
        self.assertEqual(len(self.history.detections), 1)
        self.assertEqual(self.history.detections[0], result)
    
    def test_add_multiple_detections(self):
        """Test adding multiple detections to history"""
        result1 = DetectionResult('english', 0.85, self.timestamp1, 3.0, 'microphone')
        result2 = DetectionResult('spanish', 0.78, self.timestamp2, 3.0, 'microphone')
        result3 = DetectionResult('french', 0.92, self.timestamp3, 3.0, 'file')
        
        self.history.add_detection(result1)
        self.history.add_detection(result2)
        self.history.add_detection(result3)
        
        self.assertEqual(len(self.history.detections), 3)
        self.assertEqual(self.history.detections[0], result1)
        self.assertEqual(self.history.detections[1], result2)
        self.assertEqual(self.history.detections[2], result3)
    
    def test_get_recent_with_empty_history(self):
        """Test get_recent returns empty list when history is empty"""
        recent = self.history.get_recent(3)
        
        self.assertIsInstance(recent, list)
        self.assertEqual(len(recent), 0)
    
    def test_get_recent_with_fewer_items_than_requested(self):
        """Test get_recent returns all items when fewer than n items exist"""
        result1 = DetectionResult('english', 0.85, self.timestamp1, 3.0, 'microphone')
        result2 = DetectionResult('spanish', 0.78, self.timestamp2, 3.0, 'microphone')
        
        self.history.add_detection(result1)
        self.history.add_detection(result2)
        
        recent = self.history.get_recent(5)
        
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0], result1)
        self.assertEqual(recent[1], result2)
    
    def test_get_recent_with_exact_number_of_items(self):
        """Test get_recent returns correct number when exactly n items exist"""
        result1 = DetectionResult('english', 0.85, self.timestamp1, 3.0, 'microphone')
        result2 = DetectionResult('spanish', 0.78, self.timestamp2, 3.0, 'microphone')
        result3 = DetectionResult('french', 0.92, self.timestamp3, 3.0, 'file')
        
        self.history.add_detection(result1)
        self.history.add_detection(result2)
        self.history.add_detection(result3)
        
        recent = self.history.get_recent(3)
        
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0], result1)
        self.assertEqual(recent[1], result2)
        self.assertEqual(recent[2], result3)
    
    def test_get_recent_with_more_items_than_requested(self):
        """Test get_recent returns only last n items when more exist"""
        result1 = DetectionResult('english', 0.85, self.timestamp1, 3.0, 'microphone')
        result2 = DetectionResult('spanish', 0.78, self.timestamp2, 3.0, 'microphone')
        result3 = DetectionResult('french', 0.92, self.timestamp3, 3.0, 'file')
        result4 = DetectionResult('german', 0.88, self.timestamp4, 3.0, 'microphone')
        result5 = DetectionResult('serbian', 0.95, self.timestamp5, 3.0, 'file')
        
        self.history.add_detection(result1)
        self.history.add_detection(result2)
        self.history.add_detection(result3)
        self.history.add_detection(result4)
        self.history.add_detection(result5)
        
        recent = self.history.get_recent(3)
        
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0], result3)
        self.assertEqual(recent[1], result4)
        self.assertEqual(recent[2], result5)
    
    def test_get_recent_default_parameter(self):
        """Test get_recent uses default n=3 when not specified"""
        for i in range(5):
            result = DetectionResult(
                f'language{i}',
                0.8 + i * 0.02,
                self.timestamp1,
                3.0,
                'microphone'
            )
            self.history.add_detection(result)
        
        recent = self.history.get_recent()
        
        self.assertEqual(len(recent), 3)
    
    def test_clear_empty_history(self):
        """Test clear on empty history does nothing"""
        self.history.clear()
        
        self.assertEqual(len(self.history.detections), 0)
    
    def test_clear_removes_all_detections(self):
        """Test clear removes all detections from history"""
        result1 = DetectionResult('english', 0.85, self.timestamp1, 3.0, 'microphone')
        result2 = DetectionResult('spanish', 0.78, self.timestamp2, 3.0, 'microphone')
        result3 = DetectionResult('french', 0.92, self.timestamp3, 3.0, 'file')
        
        self.history.add_detection(result1)
        self.history.add_detection(result2)
        self.history.add_detection(result3)
        
        self.assertEqual(len(self.history.detections), 3)
        
        self.history.clear()
        
        self.assertEqual(len(self.history.detections), 0)
    
    def test_clear_allows_new_detections_after(self):
        """Test that new detections can be added after clearing"""
        result1 = DetectionResult('english', 0.85, self.timestamp1, 3.0, 'microphone')
        self.history.add_detection(result1)
        
        self.history.clear()
        
        result2 = DetectionResult('spanish', 0.78, self.timestamp2, 3.0, 'microphone')
        self.history.add_detection(result2)
        
        self.assertEqual(len(self.history.detections), 1)
        self.assertEqual(self.history.detections[0], result2)
    
    def test_export_to_file_empty_history(self):
        """Test export_to_file creates valid file with empty history"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            filepath = f.name
        
        try:
            self.history.export_to_file(filepath)
            
            self.assertTrue(os.path.exists(filepath))
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertIn('Language Detection Session History', content)
            self.assertIn('No detections recorded', content)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_export_to_file_with_detections(self):
        """Test export_to_file creates valid file with detection data"""
        result1 = DetectionResult('english', 0.85, self.timestamp1, 3.0, 'microphone')
        result2 = DetectionResult('spanish', 0.78, self.timestamp2, 3.5, 'microphone')
        result3 = DetectionResult('french', 0.92, self.timestamp3, 4.0, 'file')
        
        self.history.add_detection(result1)
        self.history.add_detection(result2)
        self.history.add_detection(result3)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            filepath = f.name
        
        try:
            self.history.export_to_file(filepath)
            
            self.assertTrue(os.path.exists(filepath))
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check header
            self.assertIn('Language Detection Session History', content)
            self.assertIn('Total Detections: 3', content)
            
            # Check session times
            self.assertIn('Session Start: 2024-01-15 14:30:00', content)
            self.assertIn('Session End: 2024-01-15 14:30:06', content)
            
            # Check detection details
            self.assertIn('Detection #1', content)
            self.assertIn('English', content)
            self.assertIn('85.00%', content)
            self.assertIn('3.00s', content)
            self.assertIn('Microphone', content)
            
            self.assertIn('Detection #2', content)
            self.assertIn('Spanish', content)
            self.assertIn('78.00%', content)
            self.assertIn('3.50s', content)
            
            self.assertIn('Detection #3', content)
            self.assertIn('French', content)
            self.assertIn('92.00%', content)
            self.assertIn('4.00s', content)
            self.assertIn('File', content)
            
            self.assertIn('End of Report', content)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_export_to_file_creates_directory_if_needed(self):
        """Test export_to_file works with nested directory paths"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'subdir', 'history.txt')
            
            # Create the subdirectory
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            result = DetectionResult('english', 0.85, self.timestamp1, 3.0, 'microphone')
            self.history.add_detection(result)
            
            self.history.export_to_file(filepath)
            
            self.assertTrue(os.path.exists(filepath))
    
    def test_export_to_file_overwrites_existing(self):
        """Test export_to_file overwrites existing file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            filepath = f.name
            f.write('Old content')
        
        try:
            result = DetectionResult('english', 0.85, self.timestamp1, 3.0, 'microphone')
            self.history.add_detection(result)
            
            self.history.export_to_file(filepath)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertNotIn('Old content', content)
            self.assertIn('Language Detection Session History', content)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_export_to_file_with_special_characters(self):
        """Test export_to_file handles language names with special characters"""
        result = DetectionResult('српски', 0.90, self.timestamp1, 3.0, 'microphone')
        self.history.add_detection(result)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            filepath = f.name
        
        try:
            self.history.export_to_file(filepath)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertIn('Српски', content)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == '__main__':
    unittest.main()
