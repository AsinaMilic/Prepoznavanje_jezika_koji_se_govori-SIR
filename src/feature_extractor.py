"""
Feature Extractor modul za ekstrakciju audio karakteristika.
"""

import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Koristi non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Optional
import yaml


class FeatureExtractionError(Exception):
    """Greška pri ekstrakciji karakteristika"""
    pass


class FeatureExtractor:
    """
    Klasa za ekstrakciju audio karakteristika iz zvučnog signala.
    
    Attributes:
        n_mfcc (int): Broj MFCC koeficijenata
        n_fft (int): FFT window size
        hop_length (int): Broj uzoraka između frejmova
        n_mels (int): Broj mel filterbanks
    """
    
    def __init__(self, n_mfcc: int = 40, n_fft: int = 2048, 
                 hop_length: int = 512, n_mels: int = 128):
        """
        Inicijalizuje FeatureExtractor sa parametrima za ekstrakciju.
        
        Args:
            n_mfcc: Broj MFCC koeficijenata (default: 40)
            n_fft: FFT window size (default: 2048)
            hop_length: Broj uzoraka između frejmova (default: 512)
            n_mels: Broj mel filterbanks (default: 128)
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    @classmethod
    def from_config(cls, config_path: str = 'config.yaml'):
        """
        Kreira FeatureExtractor iz konfiguracione datoteke.
        
        Args:
            config_path: Putanja do config.yaml fajla
            
        Returns:
            FeatureExtractor instanca sa parametrima iz config fajla
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            features_config = config.get('features', {})
            return cls(
                n_mfcc=features_config.get('n_mfcc', 40),
                n_fft=features_config.get('n_fft', 2048),
                hop_length=features_config.get('hop_length', 512),
                n_mels=features_config.get('n_mels', 128)
            )
        except Exception as e:
            raise FeatureExtractionError(
                f"Greška pri učitavanju konfiguracije: {str(e)}"
            )
    
    def extract_mfcc(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Ekstraktuje MFCC koeficijente iz audio signala.
        
        Args:
            signal: Audio signal kao numpy array
            sr: Sample rate signala
            
        Returns:
            MFCC matrica dimenzija (n_mfcc, time_steps)
            
        Raises:
            FeatureExtractionError: Ako signal nije validan ili je prekratak
        """
        if signal is None or len(signal) == 0:
            raise FeatureExtractionError("Signal je prazan ili None")
        
        # Provera minimalne dužine signala
        min_length = self.n_fft
        if len(signal) < min_length:
            raise FeatureExtractionError(
                f"Audio signal je prekratak ({len(signal)} uzoraka). "
                f"Minimalna dužina: {min_length} uzoraka"
            )
        
        try:
            # Ekstrakcija MFCC koeficijenata
            mfcc = librosa.feature.mfcc(
                y=signal,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            return mfcc
            
        except Exception as e:
            raise FeatureExtractionError(
                f"Greška pri ekstrakciji MFCC: {str(e)}"
            )
    
    def extract_mel_spectrogram(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Generiše mel-spektrogram iz audio signala.
        
        Args:
            signal: Audio signal kao numpy array
            sr: Sample rate signala
            
        Returns:
            Mel-spektrogram matrica dimenzija (n_mels, time_steps)
            
        Raises:
            FeatureExtractionError: Ako signal nije validan
        """
        if signal is None or len(signal) == 0:
            raise FeatureExtractionError("Signal je prazan ili None")
        
        if len(signal) < self.n_fft:
            raise FeatureExtractionError(
                f"Audio signal je prekratak ({len(signal)} uzoraka). "
                f"Minimalna dužina: {self.n_fft} uzoraka"
            )
        
        try:
            # Generisanje mel-spektrograma
            mel_spec = librosa.feature.melspectrogram(
                y=signal,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Konverzija u dB skalu
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db
            
        except Exception as e:
            raise FeatureExtractionError(
                f"Greška pri generisanju mel-spektrograma: {str(e)}"
            )
    
    def extract_spectral_features(self, signal: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Ekstraktuje spektralne karakteristike iz audio signala.
        
        Args:
            signal: Audio signal kao numpy array
            sr: Sample rate signala
            
        Returns:
            Dict sa spektralnim karakteristikama:
                - spectral_centroid: Spektralni centroid
                - spectral_rolloff: Spektralni rolloff
                - zero_crossing_rate: Zero crossing rate
                
        Raises:
            FeatureExtractionError: Ako signal nije validan
        """
        if signal is None or len(signal) == 0:
            raise FeatureExtractionError("Signal je prazan ili None")
        
        if len(signal) < self.n_fft:
            raise FeatureExtractionError(
                f"Audio signal je prekratak ({len(signal)} uzoraka). "
                f"Minimalna dužina: {self.n_fft} uzoraka"
            )
        
        try:
            # Ekstrakcija spektralnog centroida
            spectral_centroid = librosa.feature.spectral_centroid(
                y=signal,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            # Ekstrakcija spektralnog rolloff-a
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=signal,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            # Ekstrakcija zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y=signal,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            return {
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'zero_crossing_rate': zero_crossing_rate
            }
            
        except Exception as e:
            raise FeatureExtractionError(
                f"Greška pri ekstrakciji spektralnih karakteristika: {str(e)}"
            )
    
    def visualize_features(self, signal: np.ndarray, sr: int, 
                          features: Optional[Dict] = None,
                          save_path: Optional[str] = None):
        """
        Vizualizuje audio signal i ekstraktovane karakteristike.
        
        Prikazuje:
        - Waveform (talasni oblik signala)
        - Mel-spektrogram
        - MFCC koeficijente
        
        Args:
            signal: Audio signal kao numpy array
            sr: Sample rate signala
            features: Opciono, dict sa već ekstraktovanim karakteristikama
            save_path: Opciono, putanja za čuvanje slike
            
        Raises:
            FeatureExtractionError: Ako signal nije validan
        """
        if signal is None or len(signal) == 0:
            raise FeatureExtractionError("Signal je prazan ili None")
        
        try:
            # Ekstraktuj karakteristike ako nisu prosleđene
            if features is None:
                mfcc = self.extract_mfcc(signal, sr)
                mel_spec = self.extract_mel_spectrogram(signal, sr)
            else:
                mfcc = features.get('mfcc')
                mel_spec = features.get('mel_spectrogram')
            
            # Kreiraj figure sa 3 subplota
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # 1. Waveform
            times = np.arange(len(signal)) / sr
            axes[0].plot(times, signal, linewidth=0.5)
            axes[0].set_title('Waveform', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Vreme (s)')
            axes[0].set_ylabel('Amplituda')
            axes[0].grid(True, alpha=0.3)
            
            # 2. Mel-spektrogram
            if mel_spec is not None:
                img1 = librosa.display.specshow(
                    mel_spec,
                    sr=sr,
                    hop_length=self.hop_length,
                    x_axis='time',
                    y_axis='mel',
                    ax=axes[1],
                    cmap='viridis'
                )
                axes[1].set_title('Mel-Spektrogram', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Vreme (s)')
                axes[1].set_ylabel('Frekvencija (Hz)')
                fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')
            
            # 3. MFCC
            if mfcc is not None:
                img2 = librosa.display.specshow(
                    mfcc,
                    sr=sr,
                    hop_length=self.hop_length,
                    x_axis='time',
                    ax=axes[2],
                    cmap='coolwarm'
                )
                axes[2].set_title('MFCC Koeficijenti', fontsize=14, fontweight='bold')
                axes[2].set_xlabel('Vreme (s)')
                axes[2].set_ylabel('MFCC koeficijent')
                fig.colorbar(img2, ax=axes[2])
            
            plt.tight_layout()
            
            # Sačuvaj ili prikaži
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            raise FeatureExtractionError(
                f"Greška pri vizualizaciji karakteristika: {str(e)}"
            )
