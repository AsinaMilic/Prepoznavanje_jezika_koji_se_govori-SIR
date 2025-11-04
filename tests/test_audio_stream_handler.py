"""
Unit tests for AudioStreamHandler.

Tests cover:
- start_recording and stop_recording methods
- buffer accumulation logic
- callback invocation when chunk is ready
- thread safety
- PyAudio mocking to avoid requiring actual microphone
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
import threading
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.audio_stream_handler import AudioStreamHandler, AudioStreamError, MicrophoneError


class TestAudioStreamHandler(unittest.TestCase):
    """Test suite for AudioStreamHandler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_rate = 16000
        self.chunk_duration = 3.0
        self.callback = Mock()
        
    def tearDown(self):
        """Clean up after tests"""
        self.callback.reset_mock()
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_initialization(self, mock_pyaudio_class):
        """Test AudioStreamHandler initialization"""
        handler = AudioStreamHandler(
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            callback=self.callback
        )
        
        self.assertEqual(handler.sample_rate, self.sample_rate)
        self.assertEqual(handler.chunk_duration, self.chunk_duration)
        self.assertEqual(handler.callback, self.callback)
        self.assertEqual(handler.chunk_size, int(self.sample_rate * self.chunk_duration))
        self.assertFalse(handler.is_recording())
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_start_recording_success(self, mock_pyaudio_class):
        """Test successful start of recording"""
        # Setup mock PyAudio
        mock_pyaudio = MagicMock()
        mock_pyaudio_class.return_value = mock_pyaudio
        mock_pyaudio.get_device_count.return_value = 1
        mock_pyaudio.get_default_input_device_info.return_value = {'name': 'Test Mic'}
        
        mock_stream = MagicMock()
        mock_pyaudio.open.return_value = mock_stream
        
        handler = AudioStreamHandler(
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            callback=self.callback
        )
        
        handler.start_recording()
        
        # Verify recording started
        self.assertTrue(handler.is_recording())
        
        # Verify PyAudio was initialized correctly
        mock_pyaudio.get_device_count.assert_called_once()
        mock_pyaudio.get_default_input_device_info.assert_called_once()
        mock_pyaudio.open.assert_called_once()
        
        # Clean up
        handler.stop_recording()
        time.sleep(0.1)  # Allow thread to finish
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_start_recording_no_devices(self, mock_pyaudio_class):
        """Test start_recording raises MicrophoneError when no devices found"""
        mock_pyaudio = MagicMock()
        mock_pyaudio_class.return_value = mock_pyaudio
        mock_pyaudio.get_device_count.return_value = 0
        
        handler = AudioStreamHandler()
        
        with self.assertRaises(MicrophoneError) as context:
            handler.start_recording()
        
        self.assertIn("No audio input devices found", str(context.exception))
        self.assertFalse(handler.is_recording())
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_start_recording_no_default_device(self, mock_pyaudio_class):
        """Test start_recording raises MicrophoneError when no default device"""
        mock_pyaudio = MagicMock()
        mock_pyaudio_class.return_value = mock_pyaudio
        mock_pyaudio.get_device_count.return_value = 1
        mock_pyaudio.get_default_input_device_info.side_effect = OSError("No default device")
        
        handler = AudioStreamHandler()
        
        with self.assertRaises(MicrophoneError) as context:
            handler.start_recording()
        
        self.assertIn("No default input device found", str(context.exception))
        self.assertFalse(handler.is_recording())
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_start_recording_already_recording(self, mock_pyaudio_class):
        """Test start_recording raises error when already recording"""
        mock_pyaudio = MagicMock()
        mock_pyaudio_class.return_value = mock_pyaudio
        mock_pyaudio.get_device_count.return_value = 1
        mock_pyaudio.get_default_input_device_info.return_value = {'name': 'Test Mic'}
        mock_pyaudio.open.return_value = MagicMock()
        
        handler = AudioStreamHandler()
        handler.start_recording()
        
        with self.assertRaises(AudioStreamError) as context:
            handler.start_recording()
        
        self.assertIn("already in progress", str(context.exception))
        
        handler.stop_recording()
        time.sleep(0.1)
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_stop_recording(self, mock_pyaudio_class):
        """Test stop_recording stops the recording"""
        mock_pyaudio = MagicMock()
        mock_pyaudio_class.return_value = mock_pyaudio
        mock_pyaudio.get_device_count.return_value = 1
        mock_pyaudio.get_default_input_device_info.return_value = {'name': 'Test Mic'}
        
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_pyaudio.open.return_value = mock_stream
        
        handler = AudioStreamHandler()
        handler.start_recording()
        self.assertTrue(handler.is_recording())
        
        handler.stop_recording()
        time.sleep(0.1)  # Allow thread to finish
        
        self.assertFalse(handler.is_recording())
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pyaudio.terminate.assert_called_once()
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_stop_recording_when_not_recording(self, mock_pyaudio_class):
        """Test stop_recording does nothing when not recording"""
        handler = AudioStreamHandler()
        
        # Should not raise any exception
        handler.stop_recording()
        self.assertFalse(handler.is_recording())
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_buffer_accumulation_and_callback(self, mock_pyaudio_class):
        """Test buffer accumulation and callback invocation when chunk is ready"""
        mock_pyaudio = MagicMock()
        mock_pyaudio_class.return_value = mock_pyaudio
        mock_pyaudio.get_device_count.return_value = 1
        mock_pyaudio.get_default_input_device_info.return_value = {'name': 'Test Mic'}
        
        # Create mock stream that returns audio data
        mock_stream = MagicMock()
        frames_per_buffer = 1024
        
        # Generate fake audio data (int16 format)
        fake_audio = np.random.randint(-32768, 32767, frames_per_buffer, dtype=np.int16)
        mock_stream.read.return_value = fake_audio.tobytes()
        mock_pyaudio.open.return_value = mock_stream
        
        callback = Mock()
        handler = AudioStreamHandler(
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            callback=callback
        )
        
        handler.start_recording()
        
        # Calculate how many reads needed to fill one chunk
        chunk_size = int(self.sample_rate * self.chunk_duration)
        reads_needed = (chunk_size // frames_per_buffer) + 2
        
        # Wait for callback to be invoked
        max_wait = 5.0  # seconds
        start_time = time.time()
        while callback.call_count == 0 and (time.time() - start_time) < max_wait:
            time.sleep(0.1)
        
        handler.stop_recording()
        time.sleep(0.1)
        
        # Verify callback was called
        self.assertGreater(callback.call_count, 0, "Callback should have been invoked")
        
        # Verify callback arguments
        call_args = callback.call_args
        audio_chunk, sample_rate = call_args[0]
        
        self.assertIsInstance(audio_chunk, np.ndarray)
        self.assertEqual(sample_rate, self.sample_rate)
        self.assertEqual(audio_chunk.dtype, np.float32)
        self.assertEqual(len(audio_chunk), chunk_size)
        
        # Verify audio is normalized to [-1.0, 1.0]
        self.assertGreaterEqual(audio_chunk.min(), -1.0)
        self.assertLessEqual(audio_chunk.max(), 1.0)
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_thread_safety(self, mock_pyaudio_class):
        """Test thread safety of buffer operations"""
        mock_pyaudio = MagicMock()
        mock_pyaudio_class.return_value = mock_pyaudio
        mock_pyaudio.get_device_count.return_value = 1
        mock_pyaudio.get_default_input_device_info.return_value = {'name': 'Test Mic'}
        
        mock_stream = MagicMock()
        fake_audio = np.random.randint(-32768, 32767, 1024, dtype=np.int16)
        mock_stream.read.return_value = fake_audio.tobytes()
        mock_pyaudio.open.return_value = mock_stream
        
        handler = AudioStreamHandler(
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration
        )
        
        # Start and stop recording multiple times from different threads
        def start_stop_cycle():
            try:
                handler.start_recording()
                time.sleep(0.05)
                handler.stop_recording()
            except AudioStreamError:
                pass  # Expected if already recording
        
        threads = []
        for _ in range(3):
            t = threading.Thread(target=start_stop_cycle)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=2.0)
        
        # Ensure handler is in consistent state
        time.sleep(0.2)
        self.assertFalse(handler.is_recording())
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_callback_exception_handling(self, mock_pyaudio_class):
        """Test that exceptions in callback don't crash the recording thread"""
        mock_pyaudio = MagicMock()
        mock_pyaudio_class.return_value = mock_pyaudio
        mock_pyaudio.get_device_count.return_value = 1
        mock_pyaudio.get_default_input_device_info.return_value = {'name': 'Test Mic'}
        
        mock_stream = MagicMock()
        fake_audio = np.random.randint(-32768, 32767, 1024, dtype=np.int16)
        mock_stream.read.return_value = fake_audio.tobytes()
        mock_pyaudio.open.return_value = mock_stream
        
        # Callback that raises exception
        def bad_callback(audio_data, sample_rate):
            raise ValueError("Test exception in callback")
        
        handler = AudioStreamHandler(
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            callback=bad_callback
        )
        
        handler.start_recording()
        time.sleep(0.5)  # Let it run for a bit
        
        # Recording should still be active despite callback exception
        self.assertTrue(handler.is_recording())
        
        handler.stop_recording()
        time.sleep(0.1)
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_is_recording_thread_safe(self, mock_pyaudio_class):
        """Test is_recording method is thread-safe"""
        mock_pyaudio = MagicMock()
        mock_pyaudio_class.return_value = mock_pyaudio
        mock_pyaudio.get_device_count.return_value = 1
        mock_pyaudio.get_default_input_device_info.return_value = {'name': 'Test Mic'}
        mock_pyaudio.open.return_value = MagicMock()
        
        handler = AudioStreamHandler()
        
        results = []
        
        def check_recording():
            for _ in range(100):
                results.append(handler.is_recording())
                time.sleep(0.001)
        
        # Start checking from multiple threads
        threads = [threading.Thread(target=check_recording) for _ in range(3)]
        for t in threads:
            t.start()
        
        # Start and stop recording while threads are checking
        handler.start_recording()
        time.sleep(0.1)
        handler.stop_recording()
        
        for t in threads:
            t.join(timeout=2.0)
        
        # All results should be boolean (no race conditions)
        self.assertTrue(all(isinstance(r, bool) for r in results))
    
    @patch('src.audio_stream_handler.pyaudio.PyAudio')
    def test_buffer_cleared_on_stop(self, mock_pyaudio_class):
        """Test that buffer is cleared when recording stops"""
        mock_pyaudio = MagicMock()
        mock_pyaudio_class.return_value = mock_pyaudio
        mock_pyaudio.get_device_count.return_value = 1
        mock_pyaudio.get_default_input_device_info.return_value = {'name': 'Test Mic'}
        
        mock_stream = MagicMock()
        fake_audio = np.random.randint(-32768, 32767, 1024, dtype=np.int16)
        mock_stream.read.return_value = fake_audio.tobytes()
        mock_pyaudio.open.return_value = mock_stream
        
        handler = AudioStreamHandler()
        handler.start_recording()
        time.sleep(0.2)  # Let some data accumulate
        handler.stop_recording()
        time.sleep(0.1)
        
        # Buffer should be empty after stop
        self.assertEqual(len(handler._buffer), 0)


if __name__ == '__main__':
    unittest.main()
