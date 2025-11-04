"""
Audio Stream Handler for real-time audio capture and processing.
"""

import numpy as np
import pyaudio
import threading
from typing import Callable, Optional
import time


class AudioStreamError(Exception):
    """Base exception for audio stream errors"""
    pass


class MicrophoneError(AudioStreamError):
    """Exception raised when microphone is not found or not accessible"""
    pass


class AudioStreamHandler:
    """
    Handles real-time audio capture from microphone with buffering and chunk processing.
    
    Captures audio in a background thread and invokes a callback function when
    a complete audio chunk is ready for processing.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_duration: float = 3.0,
                 callback: Optional[Callable[[np.ndarray, int], None]] = None):
        """
        Initialize AudioStreamHandler.
        
        Args:
            sample_rate: Sample rate for audio capture in Hz (default: 16000)
            chunk_duration: Duration of audio chunks in seconds (default: 3.0)
            callback: Function to call when audio chunk is ready.
                     Signature: callback(audio_data: np.ndarray, sample_rate: int)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.callback = callback
        
        # Calculate chunk size in frames
        self.chunk_size = int(sample_rate * chunk_duration)
        
        # PyAudio configuration
        self.format = pyaudio.paInt16
        self.channels = 1
        self.frames_per_buffer = 1024
        
        # State management
        self._recording = False
        self._stream = None
        self._audio_interface = None
        self._recording_thread = None
        self._buffer = []
        self._lock = threading.Lock()
        
    def start_recording(self):
        """
        Start recording audio from the microphone.
        
        Raises:
            MicrophoneError: If microphone is not found or not accessible
            AudioStreamError: If recording is already in progress
        """
        if self._recording:
            raise AudioStreamError("Recording is already in progress")
        
        try:
            # Initialize PyAudio
            self._audio_interface = pyaudio.PyAudio()
            
            # Check if microphone is available
            device_count = self._audio_interface.get_device_count()
            if device_count == 0:
                raise MicrophoneError("No audio input devices found")
            
            # Find default input device
            default_input = None
            try:
                default_input = self._audio_interface.get_default_input_device_info()
            except OSError as e:
                raise MicrophoneError(f"No default input device found: {e}")
            
            # Open audio stream
            try:
                self._stream = self._audio_interface.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.frames_per_buffer,
                    stream_callback=None
                )
            except OSError as e:
                raise MicrophoneError(f"Failed to open audio stream: {e}")
            
            # Clear buffer and start recording
            with self._lock:
                self._buffer = []
                self._recording = True
            
            # Start recording thread
            self._recording_thread = threading.Thread(target=self._record_audio, daemon=True)
            self._recording_thread.start()
            
        except Exception as e:
            # Clean up on error
            self._cleanup()
            if isinstance(e, (MicrophoneError, AudioStreamError)):
                raise
            raise AudioStreamError(f"Failed to start recording: {e}")
    
    def stop_recording(self):
        """
        Stop recording audio from the microphone.
        
        Ensures proper cleanup of audio stream and recording thread.
        """
        if not self._recording:
            return
        
        # Signal recording thread to stop
        with self._lock:
            self._recording = False
        
        # Wait for recording thread to finish
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=2.0)
        
        # Clean up resources
        self._cleanup()
    
    def is_recording(self) -> bool:
        """
        Check if recording is currently active.
        
        Returns:
            bool: True if recording is active, False otherwise
        """
        with self._lock:
            return self._recording
    
    def _record_audio(self):
        """
        Internal method that runs in background thread to capture audio.
        
        Continuously reads audio data from the stream, accumulates it in a buffer,
        and invokes the callback when a complete chunk is ready.
        """
        try:
            while self.is_recording():
                try:
                    # Read audio data from stream
                    audio_data = self._stream.read(self.frames_per_buffer, exception_on_overflow=False)
                    
                    # Convert bytes to numpy array
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Add to buffer
                    with self._lock:
                        self._buffer.extend(audio_array)
                        
                        # Check if we have enough data for a chunk
                        if len(self._buffer) >= self.chunk_size:
                            # Extract chunk
                            chunk = np.array(self._buffer[:self.chunk_size], dtype=np.float32)
                            
                            # Normalize to [-1.0, 1.0]
                            chunk = chunk / 32768.0
                            
                            # Clear processed data from buffer
                            self._buffer = self._buffer[self.chunk_size:]
                            
                            # Invoke callback if provided
                            if self.callback:
                                try:
                                    self.callback(chunk, self.sample_rate)
                                except Exception as e:
                                    print(f"Error in callback: {e}")
                
                except OSError as e:
                    if self.is_recording():
                        print(f"Stream error: {e}")
                        with self._lock:
                            self._recording = False
                    break
                    
        except Exception as e:
            print(f"Recording thread error: {e}")
            with self._lock:
                self._recording = False
    
    def _cleanup(self):
        """
        Clean up audio stream and PyAudio resources.
        
        Ensures proper resource cleanup even in case of errors.
        """
        try:
            if self._stream:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
                self._stream = None
        except Exception as e:
            print(f"Error closing stream: {e}")
        
        try:
            if self._audio_interface:
                self._audio_interface.terminate()
                self._audio_interface = None
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")
        
        # Clear buffer
        with self._lock:
            self._buffer = []
