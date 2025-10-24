"""
Hybrid CNN-RNN Model za klasifikaciju jezika.
Kombinuje CNN za prostornu ekstrakciju karakteristika sa RNN za temporalno modelovanje.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from typing import Tuple, Optional
import os


class HybridCnnRnnLanguageClassifier:
    """
    Hibridni CNN-RNN model za klasifikaciju jezika.
    CNN ekstraktuje prostorne karakteristike iz spektrograma,
    zatim RNN modeluje temporalne zavisnosti.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int,
                 cnn_filters: list = None, lstm_units: list = None, 
                 dropout_rate: float = 0.3, dense_units: int = 64):
        """
        Inicijalizuje Hybrid klasifikator.
        
        Args:
            input_shape: Dimenzije ulaznog spektrograma (height, width, channels)
            num_classes: Broj jezika za klasifikaciju
            cnn_filters: Lista broja CNN filtera (default: [32, 64])
            lstm_units: Lista LSTM jedinica (default: [128, 64])
            dropout_rate: Dropout rate (default: 0.3)
            dense_units: Broj neurona u dense sloju (default: 64)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cnn_filters = cnn_filters if cnn_filters is not None else [32, 64]
        self.lstm_units = lstm_units if lstm_units is not None else [128, 64]
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.model: Optional[keras.Model] = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """
        Gradi CNN-RNN hibridnu arhitekturu.
        
        Returns:
            Keras model
        """
        model_layers = []
        
        # CNN blokovi
        for i, num_filters in enumerate(self.cnn_filters):
            if i == 0:
                model_layers.append(layers.Conv2D(num_filters, (3, 3), padding='same', input_shape=self.input_shape))
            else:
                model_layers.append(layers.Conv2D(num_filters, (3, 3), padding='same'))
            
            model_layers.extend([
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25)
            ])
        
        # Reshape za RNN
        last_cnn_filters = self.cnn_filters[-1]
        model_layers.append(layers.Reshape((-1, last_cnn_filters)))
        
        # RNN blokovi
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            model_layers.extend([
                layers.Bidirectional(layers.LSTM(units, return_sequences=return_sequences)),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate)
            ])
        
        # Dense slojevi
        model_layers.extend([
            layers.Dense(self.dense_units, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model = models.Sequential(model_layers)
        
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
            features: Spektrogram karakteristike (može biti jedan uzorak ili batch)
                     Shape: (height, width, channels) ili (batch, height, width, channels)
            
        Returns:
            Verovatnoće za svaki jezik
            Shape: (num_classes,) ili (batch, num_classes)
        """
        if self.model is None:
            raise ValueError("Model nije izgrađen ili učitan.")
        
        # Ako je jedan uzorak, dodaj batch dimenziju
        if len(features.shape) == 3:
            features = np.expand_dims(features, axis=0)
            predictions = self.model.predict(features, verbose=0)
            return predictions[0]
        
        return self.model.predict(features, verbose=0)
    
    def save_model(self, path: str):
        """
        Čuva trenirani model.
        
        Args:
            path: Putanja za čuvanje modela (sa .h5 ili .keras ekstenzijom)
        """
        if self.model is None:
            raise ValueError("Model nije izgrađen ili treniran.")
        
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
