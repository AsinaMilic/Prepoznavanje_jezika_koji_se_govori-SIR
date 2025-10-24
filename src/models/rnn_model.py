"""
RNN/LSTM Model za klasifikaciju jezika iz sekvenci MFCC koeficijenata.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from typing import Tuple, Optional
import os


class RNNLanguageClassifier:
    """
    Rekurentna neuronska mreža (LSTM) za klasifikaciju jezika iz sekvenci MFCC.
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int, 
                 lstm_units: int = 128, dropout_rate: float = 0.3, dense_units: int = 64):
        """
        Inicijalizuje RNN/LSTM klasifikator.
        
        Args:
            input_shape: Dimenzije ulazne sekvence (time_steps, features)
            num_classes: Broj jezika za klasifikaciju
            lstm_units: Broj LSTM jedinica (default: 128)
            dropout_rate: Dropout rate (default: 0.3)
            dense_units: Broj neurona u dense sloju (default: 64)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.model: Optional[keras.Model] = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """
        Gradi RNN/LSTM arhitekturu.
        
        Returns:
            Keras model
        """
        model = models.Sequential([
            # Prvi Bidirectional LSTM sloj
            layers.Bidirectional(
                layers.LSTM(self.lstm_units, return_sequences=True),
                input_shape=self.input_shape
            ),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            # Drugi Bidirectional LSTM sloj
            layers.Bidirectional(layers.LSTM(self.lstm_units)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            # Dense slojevi
            layers.Dense(self.dense_units, activation='relu'),
            layers.Dropout(self.dropout_rate),
            
            # Output sloj
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32,
              learning_rate: float = 0.001,
              early_stopping_patience: int = 10) -> keras.callbacks.History:
        """
        Trenira model.
        
        Args:
            X_train: Trening podaci
            y_train: Trening labele (one-hot encoded)
            X_val: Validacioni podaci
            y_val: Validacione labele (one-hot encoded)
            epochs: Broj epoha treniranja
            batch_size: Veličina batch-a
            learning_rate: Learning rate za optimizer
            early_stopping_patience: Broj epoha za early stopping
            
        Returns:
            History objekat sa metrikama treniranja
        """
        if self.model is None:
            raise ValueError("Model nije izgrađen. Pozovite build_model() prvo.")
        
        # Kompajliranje modela
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Treniranje
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluira model na test skupu.
        
        Args:
            X_test: Test podaci
            y_test: Test labele (one-hot encoded)
            
        Returns:
            Dict sa metrikama (accuracy, loss)
        """
        if self.model is None:
            raise ValueError("Model nije izgrađen ili učitan.")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predviđa jezik za date karakteristike.
        
        Args:
            features: MFCC sekvence (može biti jedan uzorak ili batch)
                     Shape: (time_steps, features) ili (batch, time_steps, features)
            
        Returns:
            Verovatnoće za svaki jezik
            Shape: (num_classes,) ili (batch, num_classes)
        """
        if self.model is None:
            raise ValueError("Model nije izgrađen ili učitan.")
        
        # Ako je jedan uzorak, dodaj batch dimenziju
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
            predictions = self.model.predict(features, verbose=0)
            return predictions[0]  # Vrati samo prvi rezultat
        
        return self.model.predict(features, verbose=0)
    
    def save_model(self, path: str):
        """
        Čuva trenirani model.
        
        Args:
            path: Putanja za čuvanje modela (sa .h5 ili .keras ekstenzijom)
        """
        if self.model is None:
            raise ValueError("Model nije izgrađen ili treniran.")
        
        # Kreiraj direktorijum ako ne postoji
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        self.model.save(path)
        print(f"Model sačuvan na: {path}")
    
    def load_model(self, path: str):
        """
        Učitava trenirani model.
        
        Args:
            path: Putanja do sačuvanog modela
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model nije pronađen na putanji: {path}")
        
        self.model = keras.models.load_model(path)
        print(f"Model učitan sa: {path}")
