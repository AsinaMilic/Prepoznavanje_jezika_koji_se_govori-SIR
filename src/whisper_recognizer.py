"""
Whisper-based language recognition module.
Uses OpenAI Whisper model for language detection from audio.
"""

import numpy as np
from typing import List, Tuple
import whisper
import librosa


# Custom exception classes
class WhisperRecognizerError(Exception):
    """Base exception for Whisper recognizer errors"""
    pass


class ModelLoadError(WhisperRecognizerError):
    """Exception raised when Whisper model fails to load"""
    pass


class WhisperLanguageRecognizer:
    """
    Language recognizer using OpenAI Whisper model.
    
    Supports language detection from audio files and numpy audio data.
    """
    
    # Language code mapping from Whisper codes to readable names
    LANGUAGE_MAP = {
        'en': 'english',
        'sr': 'serbian',
        'de': 'german',
        'es': 'spanish',
        'fr': 'french',
        'it': 'italian',
        'pt': 'portuguese',
        'ru': 'russian',
        'zh': 'chinese',
        'ja': 'japanese',
        'ko': 'korean',
        'ar': 'arabic',
        'hi': 'hindi',
        'tr': 'turkish',
        'pl': 'polish',
        'nl': 'dutch',
        'sv': 'swedish',
        'da': 'danish',
        'no': 'norwegian',
        'fi': 'finnish',
    }
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize WhisperLanguageRecognizer with specified model size.
        
        Args:
            model_size: Size of Whisper model to use.
                       Options: "tiny", "base", "small", "medium", "large"
                       Default: "base" (recommended for real-time use)
        
        Raises:
            ModelLoadError: If Whisper model fails to load
        """
        self.model_size = model_size
        self.model = None
        
        try:
            self.model = whisper.load_model(model_size)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Whisper model '{model_size}'. "
                f"Please ensure openai-whisper is installed correctly. "
                f"Error: {str(e)}"
            )
    
    def recognize_from_file(self, audio_path: str, silence_threshold: float = 0.01, 
                           min_confidence: float = 0.3) -> List[Tuple[str, float]]:
        """
        Detect language from an audio file.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
            silence_threshold: RMS threshold below which audio is considered silence (default: 0.01)
            min_confidence: Minimum confidence threshold for detection (default: 0.3)
        
        Returns:
            List with single tuple containing (language_name, probability) for the detected language.
            Returns empty list if audio is silence or confidence is too low.
            Only returns one of the 5 supported languages: english, serbian, german, spanish, french
        
        Raises:
            WhisperRecognizerError: If audio file cannot be processed
        """
        try:
            # Load audio file
            audio = whisper.load_audio(audio_path)
            
            # Check for silence
            rms_energy = np.sqrt(np.mean(audio ** 2))
            if rms_energy < silence_threshold:
                return []
            
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect language
            _, probs = self.model.detect_language(mel)
            
            # Filter to only supported languages
            supported_langs = {'en', 'sr', 'de', 'es', 'fr'}
            filtered_probs = {lang: prob for lang, prob in probs.items() if lang in supported_langs}
            
            if not filtered_probs:
                return []
            
            # Get the best language
            best_lang_code = max(filtered_probs, key=filtered_probs.get)
            best_prob = filtered_probs[best_lang_code]
            
            # Check if confidence is high enough
            if best_prob < min_confidence:
                return []
            
            lang_name = self.LANGUAGE_MAP.get(best_lang_code, best_lang_code)
            return [(lang_name, float(best_prob))]
            
        except Exception as e:
            raise WhisperRecognizerError(
                f"Failed to recognize language from file '{audio_path}': {str(e)}"
            )
    
    def recognize_from_audio(self, audio_data: np.ndarray, sample_rate: int = 16000, 
                            silence_threshold: float = 0.01, min_confidence: float = 0.3) -> List[Tuple[str, float]]:
        """
        Detect language from numpy audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data (default: 16000)
            silence_threshold: RMS threshold below which audio is considered silence (default: 0.01)
            min_confidence: Minimum confidence threshold for detection (default: 0.3)
        
        Returns:
            List with single tuple containing (language_name, probability) for the detected language.
            Returns empty list if audio is silence or confidence is too low.
            Only returns one of the 5 supported languages: english, serbian, german, spanish, french
        
        Raises:
            WhisperRecognizerError: If audio data cannot be processed
        """
        try:
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
            
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize to [-1, 1] range if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Check for silence - calculate RMS (Root Mean Square) energy
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            if rms_energy < silence_threshold:
                # Audio is too quiet, likely silence
                return []
            
            # Pad or trim to 30 seconds
            audio_data = whisper.pad_or_trim(audio_data)
            
            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_data).to(self.model.device)
            
            # Detect language
            _, probs = self.model.detect_language(mel)
            
            # Filter to only supported languages
            supported_langs = {'en', 'sr', 'de', 'es', 'fr'}
            filtered_probs = {lang: prob for lang, prob in probs.items() if lang in supported_langs}
            
            if not filtered_probs:
                return []
            
            # Get the best language
            best_lang_code = max(filtered_probs, key=filtered_probs.get)
            best_prob = filtered_probs[best_lang_code]
            
            # Check if confidence is high enough
            if best_prob < min_confidence:
                return []
            
            lang_name = self.LANGUAGE_MAP.get(best_lang_code, best_lang_code)
            return [(lang_name, float(best_prob))]
            
        except Exception as e:
            raise WhisperRecognizerError(
                f"Failed to recognize language from audio data: {str(e)}"
            )
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language names.
        
        Returns:
            List of supported language names in readable format
        """
        return sorted(list(self.LANGUAGE_MAP.values()))
