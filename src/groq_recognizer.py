"""Groq Whisper API language recognition."""

import numpy as np
import requests
import tempfile
import os
from typing import List, Tuple
from pathlib import Path
import soundfile as sf


class GroqRecognizerError(Exception):
    pass


class GroqAPIError(GroqRecognizerError):
    pass


class GroqLanguageRecognizer:
    """Cloud-based language recognizer using Groq Whisper API."""
    
    LANGUAGE_MAP = {
        'english': 'english', 'serbian': 'serbian', 'croatian': 'serbian',
        'bosnian': 'serbian', 'german': 'german', 'spanish': 'spanish', 'french': 'french',
    }
    SUPPORTED = {'english', 'serbian', 'german', 'spanish', 'french'}
    
    def __init__(self, api_key: str, model: str = "whisper-large-v3"):
        if not api_key:
            raise GroqRecognizerError("API key is required")
        
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"
    
    def recognize_from_file(self, audio_path: str, silence_threshold: float = 0.01,
                           min_confidence: float = 0.3) -> List[Tuple[str, float]]:
        try:
            audio_data, _ = sf.read(audio_path)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            if np.sqrt(np.mean(audio_data ** 2)) < silence_threshold:
                return []
            
            return self._call_groq_api(audio_path, min_confidence)
        except Exception as e:
            raise GroqRecognizerError(f"Failed: {str(e)}")
    
    def recognize_from_audio(self, audio_data: np.ndarray, sample_rate: int = 16000,
                            silence_threshold: float = 0.01, min_confidence: float = 0.3) -> List[Tuple[str, float]]:
        try:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            if np.sqrt(np.mean(audio_data ** 2)) < silence_threshold:
                return []
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio_data, sample_rate)
            
            try:
                return self._call_groq_api(temp_path, min_confidence)
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        except Exception as e:
            raise GroqRecognizerError(f"Failed: {str(e)}")
    
    def _call_groq_api(self, audio_path: str, min_confidence: float) -> List[Tuple[str, float]]:
        try:
            with open(audio_path, 'rb') as f:
                files = {
                    'file': (Path(audio_path).name, f, 'audio/mpeg'),
                    'model': (None, self.model),
                    'response_format': (None, 'verbose_json'),
                }
                response = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    timeout=30
                )
            
            if response.status_code != 200:
                raise GroqAPIError(f"API failed: {response.status_code}")
            
            result = response.json()
            lang = self.LANGUAGE_MAP.get(result.get('language', '').lower())
            
            if lang and lang in self.SUPPORTED:
                return [(lang, self._calculate_confidence(result))]
            return []
            
        except requests.exceptions.RequestException as e:
            raise GroqAPIError(f"Request failed: {str(e)}")
        except Exception as e:
            raise GroqRecognizerError(f"API error: {str(e)}")
    
    def _calculate_confidence(self, result: dict) -> float:
        """Calculate confidence from API segments (avg_logprob + no_speech_prob)."""
        segments = result.get('segments', [])
        if not segments:
            return 0.75
        
        logprobs = [s['avg_logprob'] for s in segments if 'avg_logprob' in s]
        no_speech = [s['no_speech_prob'] for s in segments if 'no_speech_prob' in s]
        
        logprob_conf = max(0.0, min(1.0, 1.0 + sum(logprobs) / len(logprobs))) if logprobs else 0.75
        speech_conf = 1.0 - (sum(no_speech) / len(no_speech)) if no_speech else 0.75
        
        return max(0.5, min(0.99, logprob_conf * 0.7 + speech_conf * 0.3))
    
    def get_supported_languages(self) -> List[str]:
        return list(self.SUPPORTED)
