"""
SVM Model za klasifikaciju jezika - klasičan Machine Learning pristup.
Koristi statističke karakteristike ekstraktovane iz MFCC koeficijenata.
"""

import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class SVMLanguageClassifier:
    """
    SVM klasifikator za prepoznavanje jezika.
    Koristi statističke karakteristike (mean, std, min, max) iz MFCC-a.
    """
    
    def __init__(self, num_classes: int, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'auto'):
        """
        Inicijalizuje SVM klasifikator.
        
        Args:
            num_classes: Broj jezika za klasifikaciju
            kernel: SVM kernel ('rbf', 'linear', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient
        """
        self.num_classes = num_classes
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _extract_statistical_features(self, X: np.ndarray) -> np.ndarray:
        """
        Ekstraktuje statističke karakteristike iz MFCC sekvenci.
        
        Args:
            X: MFCC matrice shape (n_samples, n_mfcc, time_steps)
            
        Returns:
            Statistički features shape (n_samples, n_features)
        """
        features_list = []
        
        for sample in X:
            # Za svaki MFCC koeficijent, izračunaj statistike
            mean = np.mean(sample, axis=1)
            std = np.std(sample, axis=1)
            min_val = np.min(sample, axis=1)
            max_val = np.max(sample, axis=1)
            
            # Konkatenuj sve statistike
            features = np.concatenate([mean, std, min_val, max_val])
            features_list.append(features)
        
        return np.array(features_list)
    
    def build_model(self):
        """
        Gradi SVM model.
        """
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,  # Omogući probability estimates
            random_state=42
        )
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              **kwargs) -> dict:
        """
        Trenira SVM model.
        
        Args:
            X_train: Trening MFCC podaci (n_samples, n_mfcc, time_steps)
            y_train: Trening labele (n_samples,) - integer encoded
            X_val: Validacioni podaci
            y_val: Validacione labele
            **kwargs: Dodatni parametri (ignorišu se, za kompatibilnost)
            
        Returns:
            Dict sa metrikama treniranja
        """
        if self.model is None:
            self.build_model()
        
        # Ekstraktuj statističke karakteristike
        print("   Ekstrakcija statističkih karakteristika...")
        X_train_features = self._extract_statistical_features(X_train)
        X_val_features = self._extract_statistical_features(X_val)
        
        # Normalizuj features
        print("   Normalizacija features...")
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        X_val_scaled = self.scaler.transform(X_val_features)
        
        # Konvertuj one-hot u integer labels ako je potrebno
        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
        if len(y_val.shape) > 1:
            y_val = np.argmax(y_val, axis=1)
        
        # Treniraj SVM
        print("   Treniranje SVM modela...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluiraj na train i val skupu
        train_accuracy = self.model.score(X_train_scaled, y_train)
        val_accuracy = self.model.score(X_val_scaled, y_val)
        
        self.is_trained = True
        
        print(f"   Train Accuracy: {train_accuracy:.4f}")
        print(f"   Validation Accuracy: {val_accuracy:.4f}")
        
        # Vrati dummy history za kompatibilnost
        history = type('History', (), {
            'history': {
                'accuracy': [train_accuracy],
                'val_accuracy': [val_accuracy],
                'loss': [0.0],
                'val_loss': [0.0]
            }
        })()
        
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluira model na test skupu.
        
        Args:
            X_test: Test MFCC podaci
            y_test: Test labele
            
        Returns:
            Dict sa metrikama (accuracy, loss)
        """
        if not self.is_trained:
            raise ValueError("Model nije treniran.")
        
        # Ekstraktuj i normalizuj features
        X_test_features = self._extract_statistical_features(X_test)
        X_test_scaled = self.scaler.transform(X_test_features)
        
        # Konvertuj one-hot u integer labels ako je potrebno
        if len(y_test.shape) > 1:
            y_test = np.argmax(y_test, axis=1)
        
        # Evaluiraj
        accuracy = self.model.score(X_test_scaled, y_test)
        
        return {
            'loss': 0.0,  # SVM nema loss u klasičnom smislu
            'accuracy': float(accuracy)
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predviđa jezik za date MFCC karakteristike.
        
        Args:
            features: MFCC matrica (n_mfcc, time_steps) ili (batch, n_mfcc, time_steps)
            
        Returns:
            Verovatnoće za svaki jezik
        """
        if not self.is_trained:
            raise ValueError("Model nije treniran.")
        
        # Ako je jedan uzorak, dodaj batch dimenziju
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
            single_sample = True
        else:
            single_sample = False
        
        # Ekstraktuj i normalizuj features
        features_extracted = self._extract_statistical_features(features)
        features_scaled = self.scaler.transform(features_extracted)
        
        # Predvidi verovatnoće
        probabilities = self.model.predict_proba(features_scaled)
        
        if single_sample:
            return probabilities[0]
        return probabilities
    
    def save_model(self, path: str):
        """
        Čuva trenirani SVM model i scaler.
        
        Args:
            path: Putanja za čuvanje modela (sa .pkl ekstenzijom)
        """
        if not self.is_trained:
            raise ValueError("Model nije treniran.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Sačuvaj model i scaler zajedno
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'num_classes': self.num_classes,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model sačuvan na: {path}")
    
    def load_model(self, path: str):
        """
        Učitava trenirani SVM model.
        
        Args:
            path: Putanja do sačuvanog modela
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model nije pronađen na putanji: {path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.num_classes = model_data['num_classes']
        self.kernel = model_data['kernel']
        self.C = model_data['C']
        self.gamma = model_data.get('gamma', 'auto')  # Backward compatibility
        self.is_trained = True
        
        print(f"Model učitan sa: {path}")
