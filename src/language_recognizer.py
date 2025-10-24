"""
Language Recognizer modul - glavni interfejs za prepoznavanje jezika iz audio zapisa.
"""

import os
import time
import pickle
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from src.audio_processor import AudioProcessor, AudioProcessingError
from src.feature_extractor import FeatureExtractor, FeatureExtractionError
from src.dataset_builder import DatasetBuilder


class LanguageRecognizerError(Exception):
    """Greška pri prepoznavanju jezika"""
    pass


class LanguageRecognizer:
    """
    Glavni interfejs za prepoznavanje jezika iz novih audio zapisa.
    
    Attributes:
        model: Trenirani model (CNN ili RNN)
        model_type: Tip modela ('cnn' ili 'rnn')
        label_encoder: LabelEncoder za dekodiranje labela
        audio_processor: AudioProcessor instanca
        feature_extractor: FeatureExtractor instanca
        max_length: Maksimalna dužina sekvence za padding/truncation
    """
    
    def __init__(self, model_path: str, label_encoder_path: str,
                 model_type: str = 'cnn', max_length: int = 100):
        """
        Inicijalizuje LanguageRecognizer.
        
        Args:
            model_path: Putanja do treniranog modela
            label_encoder_path: Putanja do label encoder pickle fajla
            model_type: 'cnn' ili 'rnn'
            max_length: Maksimalna dužina sekvence
            
        Raises:
            LanguageRecognizerError: Ako model ili label encoder ne mogu biti učitani
        """
        if model_type not in ['cnn', 'rnn', 'wav2vec', 'hybrid_cnn_rnn', 'svm']:
            raise LanguageRecognizerError(
                f"Nepoznat tip modela: {model_type}. Podržani tipovi: 'cnn', 'rnn', 'wav2vec', 'hybrid_cnn_rnn', 'svm'"
            )
        
        self.model_type = model_type
        self.max_length = max_length
        
        # Učitaj label encoder
        try:
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"Label encoder učitan: {list(self.label_encoder.classes_)}")
        except Exception as e:
            raise LanguageRecognizerError(
                f"Greška pri učitavanju label encoder-a: {str(e)}"
            )
        
        # Učitaj model
        try:
            if model_type == 'cnn':
                from src.models.cnn_model import CNNLanguageClassifier
                self.model = CNNLanguageClassifier(
                    input_shape=(128, 100, 1),
                    num_classes=len(self.label_encoder.classes_)
                )
                self.model.load_model(model_path)
            elif model_type == 'rnn':
                from src.models.rnn_model import RNNLanguageClassifier
                self.model = RNNLanguageClassifier(
                    input_shape=(100, 40),
                    num_classes=len(self.label_encoder.classes_)
                )
                self.model.load_model(model_path)
            elif model_type == 'wav2vec':
                from src.models.wav2vec_model import Wav2VecLanguageClassifier
                self.model = Wav2VecLanguageClassifier(
                    input_shape=(100, 40),
                    num_classes=len(self.label_encoder.classes_)
                )
                self.model.load_model(model_path)
            elif model_type == 'hybrid_cnn_rnn':
                from src.models.hybrid_cnn_rnn_model import HybridCnnRnnLanguageClassifier
                self.model = HybridCnnRnnLanguageClassifier(
                    input_shape=(128, 100, 1),
                    num_classes=len(self.label_encoder.classes_)
                )
                self.model.load_model(model_path)
            else:  # svm
                from src.models.svm_model import SVMLanguageClassifier
                self.model = SVMLanguageClassifier(
                    num_classes=len(self.label_encoder.classes_)
                )
                self.model.load_model(model_path)
            print(f"Model učitan: {model_type.upper()}")
        except Exception as e:
            raise LanguageRecognizerError(
                f"Greška pri učitavanju modela: {str(e)}"
            )
        
        # Inicijalizuj audio processor i feature extractor
        self.audio_processor = AudioProcessor(target_sr=16000)
        self.feature_extractor = FeatureExtractor.from_config()
    
    def _prepare_features(self, audio_path: str) -> np.ndarray:
        """
        Priprema karakteristike iz audio fajla.
        
        Args:
            audio_path: Putanja do audio fajla
            
        Returns:
            Pripremljene karakteristike za model
            
        Raises:
            LanguageRecognizerError: Ako karakteristike ne mogu biti ekstraktovane
        """
        try:
            # Učitaj i preprocesiraj audio
            signal, sr = self.audio_processor.load_audio(audio_path)
            signal = self.audio_processor.preprocess(signal, sr)
            
            # Ekstraktuj karakteristike u zavisnosti od tipa modela
            if self.model_type in ['cnn', 'hybrid_cnn_rnn']:
                # Za CNN i Hybrid CNN-RNN koristimo mel-spektrogram
                features = self.feature_extractor.extract_mel_spectrogram(
                    signal, self.audio_processor.target_sr
                )
            else:  # rnn, wav2vec ili svm
                # Za RNN, Wav2Vec i SVM koristimo MFCC
                features = self.feature_extractor.extract_mfcc(
                    signal, self.audio_processor.target_sr
                )
            
            # Normalizuj dužinu sekvence
            dataset_builder = DatasetBuilder(
                data_dir='.',  # Dummy path
                audio_processor=self.audio_processor,
                feature_extractor=self.feature_extractor
            )
            features = dataset_builder.pad_or_truncate(features, self.max_length)
            
            # Pripremi za model
            if self.model_type in ['cnn', 'hybrid_cnn_rnn']:
                # CNN i Hybrid CNN-RNN očekuju (height, width, channels)
                features = np.expand_dims(features, axis=-1)
            elif self.model_type == 'svm':
                # SVM očekuje (n_mfcc, time_steps) - originalni MFCC format
                pass  # Već je u pravom formatu
            else:  # rnn ili wav2vec
                # RNN i Wav2Vec očekuju (time_steps, features)
                features = features.T  # Transponuj (features, time) -> (time, features)
            
            return features
            
        except (AudioProcessingError, FeatureExtractionError) as e:
            raise LanguageRecognizerError(
                f"Greška pri pripremi karakteristika: {str(e)}"
            )
    
    def recognize(self, audio_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Prepoznaje jezik iz audio zapisa.
        
        Args:
            audio_path: Putanja do audio fajla
            top_k: Broj top jezika za vraćanje
            
        Returns:
            Lista (language, probability) sortirana po verovatnoći (opadajuće)
            
        Raises:
            LanguageRecognizerError: Ako prepoznavanje ne uspe
        """
        if not os.path.exists(audio_path):
            raise LanguageRecognizerError(f"Audio fajl ne postoji: {audio_path}")
        
        try:
            # Meri vreme obrade
            start_time = time.time()
            
            # Pripremi karakteristike
            features = self._prepare_features(audio_path)
            
            # Predvidi jezik
            probabilities = self.model.predict(features)
            
            # Izračunaj vreme obrade
            processing_time = time.time() - start_time
            
            # Sortiraj jezike po verovatnoći
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                language = self.label_encoder.classes_[idx]
                probability = float(probabilities[idx])
                results.append((language, probability))
            
            # Dodaj vreme obrade u rezultate (kao dodatni info)
            self._last_processing_time = processing_time
            
            return results
            
        except LanguageRecognizerError:
            raise
        except Exception as e:
            raise LanguageRecognizerError(
                f"Greška pri prepoznavanju jezika: {str(e)}"
            )
    
    def batch_recognize(self, audio_paths: List[str], top_k: int = 3) -> List[Dict]:
        """
        Prepoznaje jezike iz više audio zapisa.
        
        Args:
            audio_paths: Lista putanja do audio fajlova
            top_k: Broj top jezika za vraćanje
            
        Returns:
            Lista dict-ova sa rezultatima za svaki fajl:
                - audio_path: Putanja do fajla
                - predictions: Lista (language, probability)
                - processing_time: Vreme obrade u sekundama
                - success: Da li je prepoznavanje uspelo
                - error: Poruka greške (ako nije uspelo)
        """
        results = []
        
        for audio_path in audio_paths:
            result = {
                'audio_path': audio_path,
                'predictions': [],
                'processing_time': 0.0,
                'success': False,
                'error': None
            }
            
            try:
                # Meri vreme obrade
                start_time = time.time()
                
                # Prepoznaj jezik
                predictions = self.recognize(audio_path, top_k=top_k)
                
                # Izračunaj vreme obrade
                processing_time = time.time() - start_time
                
                result['predictions'] = predictions
                result['processing_time'] = processing_time
                result['success'] = True
                
            except LanguageRecognizerError as e:
                result['error'] = str(e)
                result['success'] = False
            
            results.append(result)
        
        return results
    
    def get_last_processing_time(self) -> float:
        """
        Vraća vreme obrade poslednjeg recognize() poziva.
        
        Returns:
            Vreme obrade u sekundama
        """
        return getattr(self, '_last_processing_time', 0.0)
    
    def get_supported_languages(self) -> List[str]:
        """
        Vraća listu podržanih jezika.
        
        Returns:
            Lista naziva jezika
        """
        return list(self.label_encoder.classes_)
