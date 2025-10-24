"""
Audio Processor modul za učitavanje i preprocesiranje audio zapisa.
"""

import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Dict
import os


class AudioProcessingError(Exception):
    """Greška pri obradi audio fajla"""
    pass


class AudioProcessor:
    """
    Klasa za učitavanje i preprocesiranje audio zapisa.
    
    Attributes:
        target_sr (int): Ciljni sample rate za normalizaciju (default: 16000 Hz)
    """
    
    def __init__(self, target_sr: int = 16000):
        """
        Inicijalizuje AudioProcessor sa ciljnim sample rate-om.
        
        Args:
            target_sr: Ciljni sample rate za normalizaciju audio signala
        """
        self.target_sr = target_sr
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Učitava audio fajl i vraća signal i sample rate.
        
        Podržani formati: WAV, MP3, FLAC
        
        Args:
            file_path: Putanja do audio fajla
            
        Returns:
            Tuple[signal, sample_rate]: Audio signal kao numpy array i sample rate
            
        Raises:
            AudioProcessingError: Ako fajl ne postoji, nije čitljiv ili je oštećen
        """
        # Provera da li fajl postoji
        if not os.path.exists(file_path):
            raise AudioProcessingError(f"Audio fajl ne postoji: {file_path}")
        
        # Provera ekstenzije fajla
        valid_extensions = ['.wav', '.mp3', '.flac']
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in valid_extensions:
            raise AudioProcessingError(
                f"Nepodržan format fajla: {file_ext}. "
                f"Podržani formati: {', '.join(valid_extensions)}"
            )
        
        try:
            # Učitavanje audio fajla koristeći librosa
            signal, sample_rate = librosa.load(file_path, sr=None, mono=False)
            
            # Provera da li je signal prazan
            if signal is None or len(signal) == 0:
                raise AudioProcessingError(f"Audio fajl je prazan: {file_path}")
            
            return signal, sample_rate
            
        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            raise AudioProcessingError(
                f"Greška pri učitavanju audio fajla '{file_path}': {str(e)}"
            )
    
    def preprocess(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Normalizuje audio signal (konverzija u mono i resampling).
        
        Args:
            signal: Audio signal kao numpy array
            sr: Trenutni sample rate signala
            
        Returns:
            Preprocesirani signal (mono, target_sr)
            
        Raises:
            AudioProcessingError: Ako signal nije validan
        """
        if signal is None or len(signal) == 0:
            raise AudioProcessingError("Signal je prazan ili None")
        
        try:
            # Konverzija stereo u mono ako je potrebno
            if signal.ndim > 1:
                # Ako je signal stereo (2D array), konvertuj u mono
                signal = librosa.to_mono(signal)
            
            # Resampling na ciljni sample rate ako je potrebno
            if sr != self.target_sr:
                signal = librosa.resample(signal, orig_sr=sr, target_sr=self.target_sr)
            
            return signal
            
        except Exception as e:
            raise AudioProcessingError(
                f"Greška pri preprocesiranju signala: {str(e)}"
            )
    
    def get_audio_info(self, file_path: str) -> Dict[str, any]:
        """
        Vraća informacije o audio fajlu.
        
        Args:
            file_path: Putanja do audio fajla
            
        Returns:
            Dict sa informacijama:
                - duration: Trajanje u sekundama
                - sample_rate: Sample rate
                - channels: Broj kanala (1=mono, 2=stereo)
                - samples: Ukupan broj uzoraka
                
        Raises:
            AudioProcessingError: Ako fajl ne može biti pročitan
        """
        try:
            # Učitaj audio fajl
            signal, sr = self.load_audio(file_path)
            
            # Odredi broj kanala
            if signal.ndim == 1:
                channels = 1
                samples = len(signal)
            else:
                channels = signal.shape[0]
                samples = signal.shape[1]
            
            # Izračunaj trajanje
            duration = samples / sr
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'channels': channels,
                'samples': samples,
                'file_path': file_path
            }
            
        except AudioProcessingError:
            raise
        except Exception as e:
            raise AudioProcessingError(
                f"Greška pri dobijanju informacija o fajlu '{file_path}': {str(e)}"
            )
