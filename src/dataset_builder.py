"""
Dataset Builder modul za pripremu dataseta za treniranje modela.
"""

import os
import numpy as np
import yaml
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

from src.audio_processor import AudioProcessor, AudioProcessingError
from src.feature_extractor import FeatureExtractor, FeatureExtractionError


class DatasetBuilderError(Exception):
    """Greška pri pravljenju dataseta"""
    pass


class DatasetBuilder:
    """
    Klasa za pripremu dataseta za treniranje modela prepoznavanja jezika.
    
    Attributes:
        data_dir (str): Direktorijum sa audio zapisima organizovanim po jezicima
        audio_processor (AudioProcessor): Instanca za obradu audio zapisa
        feature_extractor (FeatureExtractor): Instanca za ekstrakciju karakteristika
        label_encoder (LabelEncoder): Encoder za mapiranje jezika u numeričke labele
    """
    
    def __init__(self, data_dir: str, audio_processor: AudioProcessor,
                 feature_extractor: FeatureExtractor):
        """
        Inicijalizuje DatasetBuilder.
        
        Args:
            data_dir: Direktorijum sa audio zapisima organizovanim po jezicima
            audio_processor: Instanca AudioProcessor klase
            feature_extractor: Instanca FeatureExtractor klase
            
        Raises:
            DatasetBuilderError: Ako data_dir ne postoji
        """
        if not os.path.exists(data_dir):
            raise DatasetBuilderError(f"Direktorijum ne postoji: {data_dir}")
        
        self.data_dir = data_dir
        self.audio_processor = audio_processor
        self.feature_extractor = feature_extractor
        self.label_encoder = LabelEncoder()
        
        # Učitaj konfiguraciju
        self.config = self._load_config()
    
    def _load_config(self, config_path: str = 'config.yaml') -> Dict:
        """
        Učitava konfiguraciju iz YAML fajla.
        
        Args:
            config_path: Putanja do config fajla
            
        Returns:
            Dict sa konfiguracijom
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise DatasetBuilderError(
                f"Greška pri učitavanju konfiguracije: {str(e)}"
            )
    
    def load_audio_files(self) -> List[Tuple[str, str]]:
        """
        Učitava putanje do audio fajlova i njihove labele iz direktorijuma.
        
        Očekivana struktura:
        data_dir/
            jezik1/
                audio1.wav
                audio2.wav
            jezik2/
                audio1.wav
                audio2.wav
        
        Returns:
            Lista tuple-ova (file_path, language_label)
            
        Raises:
            DatasetBuilderError: Ako nema validnih audio fajlova
        """
        audio_files = []
        valid_extensions = {'.wav', '.mp3', '.flac'}
        
        try:
            # Iteriraj kroz poddirektorijume (svaki predstavlja jezik)
            for language_dir in os.listdir(self.data_dir):
                language_path = os.path.join(self.data_dir, language_dir)
                
                # Preskoči ako nije direktorijum
                if not os.path.isdir(language_path):
                    continue
                
                # Iteriraj kroz audio fajlove u direktorijumu jezika
                for audio_file in os.listdir(language_path):
                    file_ext = os.path.splitext(audio_file)[1].lower()
                    
                    # Proveri da li je validan audio format
                    if file_ext in valid_extensions:
                        file_path = os.path.join(language_path, audio_file)
                        audio_files.append((file_path, language_dir))
            
            if len(audio_files) == 0:
                raise DatasetBuilderError(
                    f"Nisu pronađeni audio fajlovi u direktorijumu: {self.data_dir}"
                )
            
            return audio_files
            
        except DatasetBuilderError:
            raise
        except Exception as e:
            raise DatasetBuilderError(
                f"Greška pri učitavanju audio fajlova: {str(e)}"
            )
    
    def pad_or_truncate(self, features: np.ndarray, max_length: int) -> np.ndarray:
        """
        Dopunjava ili skraćuje sekvence karakteristika na fiksnu dužinu.
        
        Args:
            features: Matrica karakteristika dimenzija (n_features, time_steps)
            max_length: Ciljna dužina vremenske dimenzije
            
        Returns:
            Normalizovana matrica dimenzija (n_features, max_length)
        """
        current_length = features.shape[1]
        
        if current_length > max_length:
            # Skrati sekvence (truncate)
            return features[:, :max_length]
        elif current_length < max_length:
            # Dopuni sekvence nulama (pad)
            pad_width = max_length - current_length
            return np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Već je tačne dužine
            return features
    
    def build_dataset(self, max_length: Optional[int] = None,
                     feature_type: str = 'mfcc') -> Tuple:
        """
        Gradi dataset ekstrakcijom karakteristika iz svih audio zapisa.
        
        Args:
            max_length: Maksimalna dužina sekvence (default: iz config.yaml)
            feature_type: Tip karakteristika ('mfcc' ili 'mel_spectrogram')
            
        Returns:
            Tuple[X_train, X_val, X_test, y_train, y_val, y_test, label_encoder]
            
        Raises:
            DatasetBuilderError: Ako dataset ne može biti napravljen
        """
        # Učitaj max_length iz konfiguracije ako nije prosleđen
        if max_length is None:
            max_length = self.config['dataset']['max_sequence_length']
        
        # Učitaj audio fajlove
        audio_files = self.load_audio_files()
        print(f"Pronađeno {len(audio_files)} audio fajlova")
        
        # Liste za čuvanje karakteristika i labela
        features_list = []
        labels_list = []
        
        # Procesiranje svakog audio fajla
        skipped_files = 0
        for idx, (file_path, language) in enumerate(audio_files):
            try:
                # Učitaj i preprocesiraj audio
                signal, sr = self.audio_processor.load_audio(file_path)
                signal = self.audio_processor.preprocess(signal, sr)
                
                # Ekstraktuj karakteristike
                if feature_type == 'mfcc':
                    features = self.feature_extractor.extract_mfcc(signal, self.audio_processor.target_sr)
                elif feature_type == 'mel_spectrogram':
                    features = self.feature_extractor.extract_mel_spectrogram(signal, self.audio_processor.target_sr)
                else:
                    raise DatasetBuilderError(f"Nepoznat tip karakteristika: {feature_type}")
                
                # Normalizuj dužinu sekvence
                features = self.pad_or_truncate(features, max_length)
                
                # Dodaj u liste
                features_list.append(features)
                labels_list.append(language)
                
                # Prikaži napredak
                if (idx + 1) % 10 == 0:
                    print(f"Procesovano {idx + 1}/{len(audio_files)} fajlova")
                
            except (AudioProcessingError, FeatureExtractionError) as e:
                print(f"Preskačem fajl {file_path}: {str(e)}")
                skipped_files += 1
                continue
        
        if len(features_list) == 0:
            raise DatasetBuilderError("Nijedan audio fajl nije uspešno procesovan")
        
        print(f"Uspešno procesovano {len(features_list)} fajlova ({skipped_files} preskočeno)")
        
        # Konvertuj u numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Enkoduj labele
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Jezici u datasetu: {list(self.label_encoder.classes_)}")
        print(f"Dimenzije karakteristika: {X.shape}")
        
        # Podeli dataset na train/val/test
        train_split = self.config['dataset']['train_split']
        val_split = self.config['dataset']['val_split']
        test_split = self.config['dataset']['test_split']
        
        # Prvo podeli na train i temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded,
            test_size=(val_split + test_split),
            random_state=42,
            stratify=y_encoded
        )
        
        # Zatim podeli temp na val i test
        val_ratio = val_split / (val_split + test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            random_state=42,
            stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} uzoraka")
        print(f"Validation set: {X_val.shape[0]} uzoraka")
        print(f"Test set: {X_test.shape[0]} uzoraka")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.label_encoder
    
    def save_dataset(self, X_train, X_val, X_test, y_train, y_val, y_test,
                    output_dir: str = 'data/processed') -> None:
        """
        Čuva pripremljeni dataset u NumPy formatu.
        
        Args:
            X_train, X_val, X_test: Karakteristike za train/val/test
            y_train, y_val, y_test: Labele za train/val/test
            output_dir: Direktorijum za čuvanje dataseta
            
        Raises:
            DatasetBuilderError: Ako dataset ne može biti sačuvan
        """
        try:
            # Kreiraj output direktorijum ako ne postoji
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Sačuvaj numpy arrays
            np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
            np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
            np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
            np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
            np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
            
            # Sačuvaj label encoder
            with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            print(f"Dataset sačuvan u: {output_dir}")
            
        except Exception as e:
            raise DatasetBuilderError(
                f"Greška pri čuvanju dataseta: {str(e)}"
            )
    
    @staticmethod
    def load_saved_dataset(input_dir: str = 'data/processed') -> Tuple:
        """
        Učitava prethodno sačuvan dataset.
        
        Args:
            input_dir: Direktorijum sa sačuvanim datasetom
            
        Returns:
            Tuple[X_train, X_val, X_test, y_train, y_val, y_test, label_encoder]
            
        Raises:
            DatasetBuilderError: Ako dataset ne može biti učitan
        """
        try:
            X_train = np.load(os.path.join(input_dir, 'X_train.npy'))
            X_val = np.load(os.path.join(input_dir, 'X_val.npy'))
            X_test = np.load(os.path.join(input_dir, 'X_test.npy'))
            y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
            y_val = np.load(os.path.join(input_dir, 'y_val.npy'))
            y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
            
            with open(os.path.join(input_dir, 'label_encoder.pkl'), 'rb') as f:
                label_encoder = pickle.load(f)
            
            print(f"Dataset učitan iz: {input_dir}")
            print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
            
        except Exception as e:
            raise DatasetBuilderError(
                f"Greška pri učitavanju dataseta: {str(e)}"
            )
