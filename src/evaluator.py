"""
Evaluator modul za evaluaciju performansi modela i vizualizaciju rezultata.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import List, Dict, Tuple, Optional, Any
import os
from datetime import datetime


class ModelEvaluator:
    """
    Klasa za evaluaciju performansi modela i vizualizaciju rezultata.
    """
    
    def __init__(self, model: Any, label_encoder: Any):
        """
        Inicijalizuje evaluator.
        
        Args:
            model: Trenirani model (CNN ili RNN)
            label_encoder: LabelEncoder za dekodiranje labela
        """
        self.model = model
        self.label_encoder = label_encoder
        
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Kompletna evaluacija modela.
        
        Args:
            X_test: Test podaci
            y_test: Test labele (one-hot encoded)
            
        Returns:
            Dict sa accuracy, precision, recall, f1_score po jezicima
        """
        # Predikcije
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Globalne metrike
        accuracy = accuracy_score(y_true, y_pred)
        
        # Metrike po klasama (weighted average)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Metrike po klasama (macro average)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Metrike po svakom jeziku
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Mapiranje na nazive jezika
        class_names = self.label_encoder.classes_
        per_language_metrics = {}
        
        for idx, language in enumerate(class_names):
            per_language_metrics[language] = {
                'precision': float(precision_per_class[idx]),
                'recall': float(recall_per_class[idx]),
                'f1_score': float(f1_per_class[idx])
            }
        
        return {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_score_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_score_macro': float(f1_macro),
            'per_language': per_language_metrics,
            'num_samples': len(y_true)
        }

    def generate_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """
        Generiše confusion matrix.
        
        Args:
            X_test: Test podaci
            y_test: Test labele (one-hot encoded)
            
        Returns:
            Confusion matrix kao numpy array
        """
        # Predikcije
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Generisanje confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return cm
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: Optional[List[str]] = None,
                             save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Vizualizuje confusion matrix kao heatmap.
        
        Args:
            cm: Confusion matrix
            class_names: Lista naziva klasa (jezika). Ako nije dato, koristi label_encoder
            save_path: Putanja za čuvanje slike (opciono)
            figsize: Veličina figure
        """
        if class_names is None:
            class_names = self.label_encoder.classes_
        
        # Kreiranje figure
        plt.figure(figsize=figsize)
        
        # Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Broj uzoraka'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Stvarni jezik', fontsize=12)
        plt.xlabel('Predviđeni jezik', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Čuvanje ako je navedena putanja
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix sačuvan na: {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history: Any, save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (14, 5)):
        """
        Plotuje accuracy i loss tokom treniranja.
        
        Args:
            history: Keras History objekat ili dict sa metrikama
            save_path: Putanja za čuvanje slike (opciono)
            figsize: Veličina figure
        """
        # Ekstraktovanje metrika iz history objekta
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history
        
        # Kreiranje subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot accuracy
        axes[0].plot(history_dict['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Accuracy', fontsize=11)
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(history_dict['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Loss', fontsize=11)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Čuvanje ako je navedena putanja
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot sačuvan na: {save_path}")
        
        plt.show()

    def save_evaluation_report(self, metrics: Dict[str, Any], output_path: str,
                               model_name: str = "Model"):
        """
        Čuva izveštaj evaluacije u tekstualnom formatu.
        
        Args:
            metrics: Dict sa metrikama iz evaluate_model()
            output_path: Putanja za čuvanje izveštaja
            model_name: Naziv modela za izveštaj
        """
        # Kreiraj direktorijum ako ne postoji
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 70 + "\n")
            f.write(f"IZVEŠTAJ EVALUACIJE MODELA: {model_name}\n")
            f.write(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            # Globalne metrike
            f.write("GLOBALNE METRIKE\n")
            f.write("-" * 70 + "\n")
            f.write(f"Broj test uzoraka: {metrics['num_samples']}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n\n")
            
            f.write("Weighted Average (uzima u obzir broj uzoraka po klasi):\n")
            f.write(f"  Precision: {metrics['precision_weighted']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall_weighted']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score_weighted']:.4f}\n\n")
            
            f.write("Macro Average (jednaka težina za sve klase):\n")
            f.write(f"  Precision: {metrics['precision_macro']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall_macro']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score_macro']:.4f}\n\n")
            
            # Metrike po jezicima
            f.write("METRIKE PO JEZICIMA\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Jezik':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-" * 70 + "\n")
            
            for language, lang_metrics in metrics['per_language'].items():
                f.write(f"{language:<15} "
                       f"{lang_metrics['precision']:<12.4f} "
                       f"{lang_metrics['recall']:<12.4f} "
                       f"{lang_metrics['f1_score']:<12.4f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            
            # Interpretacija rezultata
            f.write("\nINTERPRETACIJA\n")
            f.write("-" * 70 + "\n")
            
            # Najbolji i najgori jezici po F1 score
            sorted_languages = sorted(metrics['per_language'].items(), 
                                     key=lambda x: x[1]['f1_score'], reverse=True)
            
            best_lang, best_metrics = sorted_languages[0]
            worst_lang, worst_metrics = sorted_languages[-1]
            
            f.write(f"Najbolje prepoznat jezik: {best_lang} (F1: {best_metrics['f1_score']:.4f})\n")
            f.write(f"Najslabije prepoznat jezik: {worst_lang} (F1: {worst_metrics['f1_score']:.4f})\n\n")
            
            # Opšta ocena
            if metrics['accuracy'] >= 0.90:
                ocena = "Odličan"
            elif metrics['accuracy'] >= 0.80:
                ocena = "Dobar"
            elif metrics['accuracy'] >= 0.70:
                ocena = "Zadovoljavajući"
            else:
                ocena = "Potrebno poboljšanje"
            
            f.write(f"Opšta ocena performansi: {ocena}\n")
            f.write("=" * 70 + "\n")
        
        print(f"Izveštaj evaluacije sačuvan na: {output_path}")
    
    def analyze_errors(self, X_test: np.ndarray, y_test: np.ndarray, 
                      file_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Identifikuje pogrešno klasifikovane uzorke.
        
        Args:
            X_test: Test podaci
            y_test: Test labele (one-hot encoded)
            file_paths: Lista putanja do audio fajlova (opciono)
            
        Returns:
            Lista dict sa informacijama o greškama:
            [{
                'index': int,
                'file_path': str (ako je dato),
                'true_label': str,
                'predicted_label': str,
                'confidence': float,
                'true_probability': float
            }]
        """
        # Predikcije
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Pronađi greške
        errors = []
        class_names = self.label_encoder.classes_
        
        for idx in range(len(y_true)):
            if y_true[idx] != y_pred[idx]:
                error_info = {
                    'index': idx,
                    'true_label': class_names[y_true[idx]],
                    'predicted_label': class_names[y_pred[idx]],
                    'confidence': float(y_pred_probs[idx][y_pred[idx]]),
                    'true_probability': float(y_pred_probs[idx][y_true[idx]])
                }
                
                if file_paths and idx < len(file_paths):
                    error_info['file_path'] = file_paths[idx]
                
                errors.append(error_info)
        
        # Sortiraj po confidence (najviša confidence prva - najsigurnije greške)
        errors.sort(key=lambda x: x['confidence'], reverse=True)
        
        return errors
    
    def print_error_analysis(self, errors: List[Dict[str, Any]], top_n: int = 10):
        """
        Ispisuje analizu grešaka u čitljivom formatu.
        
        Args:
            errors: Lista grešaka iz analyze_errors()
            top_n: Broj grešaka za prikaz
        """
        print("\n" + "=" * 80)
        print(f"ANALIZA GREŠAKA - Top {min(top_n, len(errors))} najsigurnijih pogrešnih predikcija")
        print("=" * 80 + "\n")
        
        if not errors:
            print("Nema grešaka! Model je savršeno klasifikovao sve uzorke.")
            return
        
        print(f"Ukupan broj grešaka: {len(errors)}\n")
        
        for i, error in enumerate(errors[:top_n], 1):
            print(f"{i}. Uzorak #{error['index']}")
            if 'file_path' in error:
                print(f"   Fajl: {error['file_path']}")
            print(f"   Stvarni jezik: {error['true_label']}")
            print(f"   Predviđeni jezik: {error['predicted_label']}")
            print(f"   Confidence: {error['confidence']:.4f} ({error['confidence']*100:.2f}%)")
            print(f"   Verovatnoća za stvarni jezik: {error['true_probability']:.4f}")
            print()
        
        print("=" * 80 + "\n")
