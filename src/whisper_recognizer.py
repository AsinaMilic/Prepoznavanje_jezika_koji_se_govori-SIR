"""Whisper-based language recognition."""

import numpy as np
from typing import List, Tuple
import whisper
import librosa


class WhisperRecognizerError(Exception):
    pass


class ModelLoadError(WhisperRecognizerError):
    pass


class WhisperLanguageRecognizer:
    """Language recognizer using OpenAI Whisper model."""
    
    LANGUAGE_MAP = {
        'en': 'english', 'sr': 'serbian', 'de': 'german',
        'es': 'spanish', 'fr': 'french',
    }
    SUPPORTED = {'en', 'sr', 'de', 'es', 'fr'}
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        try:
            self.model = whisper.load_model(model_size)
        except Exception as e:
            raise ModelLoadError(f"Failed to load Whisper model: {str(e)}")
    
    def recognize_from_file(self, audio_path: str, silence_threshold: float = 0.01, 
                           min_confidence: float = 0.3) -> List[Tuple[str, float]]:
        try:
            audio = whisper.load_audio(audio_path)
            
            if np.sqrt(np.mean(audio ** 2)) < silence_threshold:
                return []
            
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            
            filtered = {k: v for k, v in probs.items() if k in self.SUPPORTED}
            if not filtered:
                return []
            
            best_code = max(filtered, key=filtered.get)
            best_prob = filtered[best_code]
            
            if best_prob < min_confidence:
                return []
            
            return [(self.LANGUAGE_MAP[best_code], float(best_prob))]
            
        except Exception as e:
            raise WhisperRecognizerError(f"Failed to recognize from file: {str(e)}")
    
    def recognize_from_audio(self, audio_data: np.ndarray, sample_rate: int = 16000, 
                            silence_threshold: float = 0.01, min_confidence: float = 0.3) -> List[Tuple[str, float]]:
        try:
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            if np.sqrt(np.mean(audio_data ** 2)) < silence_threshold:
                return []
            
            audio_data = whisper.pad_or_trim(audio_data)
            mel = whisper.log_mel_spectrogram(audio_data).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            
            filtered = {k: v for k, v in probs.items() if k in self.SUPPORTED}
            if not filtered:
                return []
            
            best_code = max(filtered, key=filtered.get)
            best_prob = filtered[best_code]
            
            if best_prob < min_confidence:
                return []
            
            return [(self.LANGUAGE_MAP[best_code], float(best_prob))]
            
        except Exception as e:
            raise WhisperRecognizerError(f"Failed to recognize from audio: {str(e)}")
    
    def get_supported_languages(self) -> List[str]:
        return list(self.LANGUAGE_MAP.values())
