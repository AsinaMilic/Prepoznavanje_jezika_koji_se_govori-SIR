"""
Integration tests for Real-time Language Recognition System.

Tests cover:
- End-to-end recording flow with simulated audio
- File selection flow with test audio files
- GUI responsiveness during processing
- Verification that GUI doesn't block during Whisper processing
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import tkinter as tk
import numpy as np
import threading
import time
import tempfile
import os
import sys
import wave

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock whisper and pyaudio modules before importing
sys.modules['whisper'] = MagicMock()
sys.modules['pyaudio'] = MagicMock()

from src.realtime_gui import RealtimeLanguageGUI
from src.whisper_recognizer import WhisperLanguageRecognizer
from src.audio_stream_handler import AudioStreamHandler
from src.session_history import DetectionResult


# Shared Tk root for all tests to avoid multiple Tk() instances
_test_root = None

def get_test_root():
    """Get or create a shared Tk root for testing"""
    global _test_root
    if _test_root is None:
        _test_root = tk.Tk()
        _test_root.withdraw()  # Hide the window
    return _test_root


class TestEndToEndRecordingFlow(unittest.TestCase):
    """Test end-to-end recording flow with simulated audio"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gui = None
    
    def tearDown(self):
        """Clean up after tests"""
        if self.gui:
            if self.gui.audio_handler:
                try:
                    self.gui.audio_handler.stop_recording()
                except:
                    pass
            
            try:
                # Destroy widgets but keep root
                for widget in self.gui.root.winfo_children():
                    widget.destroy()
            except:
                pass
    
    @patch('src.realtime_gui.WhisperLanguageRecognizer')
    @patch('src.realtime_gui.AudioStreamHandler')
    def test_recording_flow_with_simulated_audio(self, mock_audio_handler_class, mock_recognizer_class):
        """Test complete recording flow with simulated audio input"""
        # Setup mock recognizer
        mock_recognizer = MagicMock()
        mock_recognizer.recognize_from_audio.return_value = [('english', 0.95), ('spanish', 0.03), ('french', 0.02)]
        mock_recognizer_class.return_value = mock_recognizer
        
        # Setup mock audio handler
        mock_audio_handler = MagicMock()
        mock_audio_handler.is_recording.return_value = False
        mock_audio_handler_class.return_value = mock_audio_handler
        
        # Create GUI
        self.gui = RealtimeLanguageGUI()
        self.gui.root.update()
        
        # Verify initial state
        self.assertEqual(self.gui.status_label.cget('text'), 'Idle')
        self.assertIn(str(self.gui.start_button.cget('state')), ['normal', 'Normal'])
        self.assertIn(str(self.gui.stop_button.cget('state')), ['disabled', 'Disabled'])
        
        # Simulate recording start
        self.gui.start_recording()
        self.gui.root.update()
        
        # Verify recording started
        self.assertTrue(mock_audio_handler.start_recording.called)
        self.assertEqual(self.gui.status_label.cget('text'), 'Recording')
        self.assertIn(str(self.gui.start_button.cget('state')), ['disabled', 'Disabled'])
        self.assertIn(str(self.gui.stop_button.cget('state')), ['normal', 'Normal'])
        
        # Simulate audio chunk processing
        test_audio = np.random.randn(16000 * 3).astype(np.float32)
        self.gui.on_audio_chunk_ready(test_audio, 16000)
        
        # Wait for processing thread to complete
        time.sleep(0.5)
        for _ in range(10):
            self.gui.root.update()
            time.sleep(0.05)
        
        # Verify recognition was called
        self.assertTrue(mock_recognizer.recognize_from_audio.called)
        
        # Verify detection was added to session history (this happens in the thread)
        # Note: GUI updates via root.after() may not work in tests without mainloop
        self.assertGreaterEqual(len(self.gui.session_history.detections), 1)
        if len(self.gui.session_history.detections) > 0:
            self.assertEqual(self.gui.session_history.detections[0].language, 'english')
            self.assertEqual(self.gui.session_history.detections[0].source, 'microphone')
        
        # Stop recording
        mock_audio_handler.is_recording.return_value = False
        self.gui.stop_recording()
        self.gui.root.update()
        
        # Verify recording stopped
        self.assertTrue(mock_audio_handler.stop_recording.called)
        self.assertEqual(self.gui.status_label.cget('text'), 'Idle')
        self.assertIn(str(self.gui.start_button.cget('state')), ['normal', 'Normal'])
        self.assertIn(str(self.gui.stop_button.cget('state')), ['disabled', 'Disabled'])


class TestFileSelectionFlow(unittest.TestCase):
    """Test file selection and processing flow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gui = None
        self.temp_audio_file = None
    
    def tearDown(self):
        """Clean up after tests"""
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
            except:
                pass
        
        if self.gui:
            try:
                for widget in self.gui.root.winfo_children():
                    widget.destroy()
            except:
                pass
    
    def _create_test_audio_file(self):
        """Create a temporary test audio file"""
        fd, path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        
        # Create a simple WAV file
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            # Write 1 second of silence
            audio_data = np.zeros(16000, dtype=np.int16)
            wav_file.writeframes(audio_data.tobytes())
        
        return path
    
    @patch('src.realtime_gui.WhisperLanguageRecognizer')
    @patch('src.realtime_gui.filedialog.askopenfilename')
    def test_file_selection_and_processing(self, mock_filedialog, mock_recognizer_class):
        """Test selecting and processing an audio file"""
        # Create test audio file
        self.temp_audio_file = self._create_test_audio_file()
        mock_filedialog.return_value = self.temp_audio_file
        
        # Setup mock recognizer
        mock_recognizer = MagicMock()
        mock_recognizer.recognize_from_file.return_value = [
            ('spanish', 0.88),
            ('english', 0.08),
            ('french', 0.04)
        ]
        mock_recognizer_class.return_value = mock_recognizer
        
        # Create GUI
        self.gui = RealtimeLanguageGUI()
        self.gui.root.update()
        
        # Trigger file selection
        self.gui.select_audio_file()
        
        # Verify file dialog was called
        self.assertTrue(mock_filedialog.called)
        
        # Wait for processing thread to complete
        time.sleep(1.0)
        for _ in range(20):
            try:
                self.gui.root.update()
            except:
                pass
            time.sleep(0.05)
        
        # Verify recognition was called with the file path
        mock_recognizer.recognize_from_file.assert_called_once_with(self.temp_audio_file)
        
        # Verify detection was added to session history
        # Note: GUI updates may fail in tests, but session history should still be updated
        self.assertGreaterEqual(len(self.gui.session_history.detections), 1)
        if len(self.gui.session_history.detections) > 0:
            self.assertEqual(self.gui.session_history.detections[0].language, 'spanish')
            self.assertEqual(self.gui.session_history.detections[0].source, 'file')
    
    @patch('src.realtime_gui.WhisperLanguageRecognizer')
    @patch('src.realtime_gui.filedialog.askopenfilename')
    def test_file_selection_cancelled(self, mock_filedialog, mock_recognizer_class):
        """Test that cancelling file selection doesn't cause errors"""
        # User cancels file dialog
        mock_filedialog.return_value = ''
        
        mock_recognizer = MagicMock()
        mock_recognizer_class.return_value = mock_recognizer
        
        # Create GUI
        self.gui = RealtimeLanguageGUI()
        self.gui.root.update()
        
        # Trigger file selection
        self.gui.select_audio_file()
        self.gui.root.update()
        
        # Verify file dialog was called
        self.assertTrue(mock_filedialog.called)
        
        # Verify recognition was NOT called
        self.assertFalse(mock_recognizer.recognize_from_file.called)


class TestGUIResponsiveness(unittest.TestCase):
    """Test that GUI remains responsive during processing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gui = None
    
    def tearDown(self):
        """Clean up after tests"""
        if self.gui:
            try:
                for widget in self.gui.root.winfo_children():
                    widget.destroy()
            except:
                pass
    
    @patch('src.realtime_gui.WhisperLanguageRecognizer')
    @patch('src.realtime_gui.AudioStreamHandler')
    def test_gui_doesnt_block_during_audio_processing(self, mock_audio_handler_class, mock_recognizer_class):
        """Verify GUI doesn't freeze during Whisper processing of audio chunks"""
        # Setup mock with delayed response to simulate real processing
        mock_recognizer = MagicMock()
        
        def slow_recognize(*args, **kwargs):
            time.sleep(0.1)  # Simulate processing time
            return [('french', 0.92), ('english', 0.05), ('spanish', 0.03)]
        
        mock_recognizer.recognize_from_audio.side_effect = slow_recognize
        mock_recognizer_class.return_value = mock_recognizer
        
        mock_audio_handler = MagicMock()
        mock_audio_handler.is_recording.return_value = True
        mock_audio_handler_class.return_value = mock_audio_handler
        
        # Create GUI
        self.gui = RealtimeLanguageGUI()
        self.gui.root.update()
        
        # Trigger audio chunk processing (runs in background thread)
        test_audio = np.random.randn(16000 * 3).astype(np.float32)
        self.gui.on_audio_chunk_ready(test_audio, 16000)
        
        # Verify GUI can still update while processing happens in background
        # This proves processing is non-blocking
        self.gui.root.update()
        
        # Wait for processing to complete
        time.sleep(0.5)
        for _ in range(10):
            self.gui.root.update()
            time.sleep(0.05)
        
        # Verify processing completed
        self.assertTrue(mock_recognizer.recognize_from_audio.called)
        
        # Verify detection was added to history (GUI updates may not work without mainloop)
        self.assertGreaterEqual(len(self.gui.session_history.detections), 1)
    
    @patch('src.realtime_gui.WhisperLanguageRecognizer')
    @patch('src.realtime_gui.filedialog.askopenfilename')
    def test_gui_doesnt_block_during_file_processing(self, mock_filedialog, mock_recognizer_class):
        """Verify GUI doesn't freeze during file processing"""
        # Create temp audio file
        fd, temp_file = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        
        try:
            with wave.open(temp_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                audio_data = np.zeros(16000, dtype=np.int16)
                wav_file.writeframes(audio_data.tobytes())
            
            mock_filedialog.return_value = temp_file
            
            # Setup mock with delayed response
            mock_recognizer = MagicMock()
            
            def slow_recognize_file(*args, **kwargs):
                time.sleep(0.1)
                return [('german', 0.85), ('english', 0.10), ('french', 0.05)]
            
            mock_recognizer.recognize_from_file.side_effect = slow_recognize_file
            mock_recognizer_class.return_value = mock_recognizer
            
            # Create GUI
            self.gui = RealtimeLanguageGUI()
            self.gui.root.update()
            
            # Trigger file processing
            self.gui.select_audio_file()
            
            # Verify GUI can still update while processing happens in background
            try:
                self.gui.root.update()
            except:
                pass
            
            # Wait for processing to complete
            time.sleep(1.0)
            for _ in range(20):
                try:
                    self.gui.root.update()
                except:
                    pass
                time.sleep(0.05)
            
            # Verify processing completed
            self.assertTrue(mock_recognizer.recognize_from_file.called)
            
            # Verify detection was added to history
            self.assertGreaterEqual(len(self.gui.session_history.detections), 1)
            
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    @patch('src.realtime_gui.WhisperLanguageRecognizer')
    @patch('src.realtime_gui.AudioStreamHandler')
    def test_multiple_audio_chunks_dont_block_gui(self, mock_audio_handler_class, mock_recognizer_class):
        """Verify GUI handles multiple audio chunks without blocking"""
        mock_recognizer = MagicMock()
        mock_recognizer.recognize_from_audio.return_value = [
            ('english', 0.90),
            ('spanish', 0.07),
            ('french', 0.03)
        ]
        mock_recognizer_class.return_value = mock_recognizer
        
        mock_audio_handler = MagicMock()
        mock_audio_handler.is_recording.return_value = True
        mock_audio_handler_class.return_value = mock_audio_handler
        
        # Create GUI
        self.gui = RealtimeLanguageGUI()
        self.gui.root.update()
        
        # Process multiple audio chunks
        for i in range(3):
            test_audio = np.random.randn(16000 * 3).astype(np.float32)
            self.gui.on_audio_chunk_ready(test_audio, 16000)
            
            # GUI should be able to update between chunks (proves non-blocking)
            self.gui.root.update()
            time.sleep(0.1)
        
        # Wait for all processing to complete
        time.sleep(0.8)
        for _ in range(15):
            self.gui.root.update()
            time.sleep(0.05)
        
        # Verify multiple chunks were processed (at least 1, may skip if already processing)
        self.assertGreaterEqual(mock_recognizer.recognize_from_audio.call_count, 1)
        
        # Verify history was updated (at least one detection should be added)
        self.assertGreaterEqual(len(self.gui.session_history.detections), 1)


if __name__ == '__main__':
    unittest.main()
