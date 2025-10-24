"""
Glavni script za treniranje CNN i RNN modela za prepoznavanje jezika.

Ovaj script:
1. Učitava konfiguraciju iz config.yaml
2. Inicijalizuje AudioProcessor, FeatureExtractor i DatasetBuilder
3. Priprema dataset pozivom build_dataset()
4. Trenira CNN i RNN modele
5. Čuva trenirane modele i label encoder
6. Prikazuje training history grafike
"""

import os
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.utils import to_categorical

from src.audio_processor import AudioProcessor
from src.feature_extractor import FeatureExtractor
from src.dataset_builder import DatasetBuilder
from src.models.cnn_model import CNNLanguageClassifier
from src.models.rnn_model import RNNLanguageClassifier
from src.models.wav2vec_model import Wav2VecLanguageClassifier
from src.models.hybrid_cnn_rnn_model import HybridCnnRnnLanguageClassifier
from src.models.svm_model import SVMLanguageClassifier


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Učitava konfiguraciju iz YAML fajla.
    
    Args:
        config_path: Putanja do config fajla
        
    Returns:
        Dict sa konfiguracijom
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def plot_training_history(history, model_name: str, output_dir: str = 'models'):
    """
    Prikazuje i čuva grafike training history-ja.
    
    Args:
        history: Keras History objekat
        model_name: Ime modela (za naziv fajla)
        output_dir: Direktorijum za čuvanje grafika
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy grafik
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss grafik
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sačuvaj grafik
    output_path = os.path.join(output_dir, f'{model_name}_training_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history grafik sačuvan: {output_path}")


def prepare_data_for_cnn(X_train, X_val, X_test):
    """
    Priprema podatke za CNN model (dodaje channel dimenziju).
    
    Args:
        X_train, X_val, X_test: Dataset arrays
        
    Returns:
        Transformisani arrays sa channel dimenzijom
    """
    # CNN očekuje (batch, height, width, channels)
    # MFCC je (batch, n_mfcc, time_steps), dodajemo channel dimenziju
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_val_cnn = np.expand_dims(X_val, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)
    
    return X_train_cnn, X_val_cnn, X_test_cnn


def prepare_data_for_rnn(X_train, X_val, X_test):
    """
    Priprema podatke za RNN model (transponuje dimenzije).
    
    Args:
        X_train, X_val, X_test: Dataset arrays
        
    Returns:
        Transformisani arrays za RNN
    """
    # RNN očekuje (batch, time_steps, features)
    # MFCC je (batch, n_mfcc, time_steps), transponujemo
    X_train_rnn = np.transpose(X_train, (0, 2, 1))
    X_val_rnn = np.transpose(X_val, (0, 2, 1))
    X_test_rnn = np.transpose(X_test, (0, 2, 1))
    
    return X_train_rnn, X_val_rnn, X_test_rnn


def main(args):
    """
    Glavna funkcija za treniranje modela.
    """
    print("=" * 70)
    print("SISTEM ZA PREPOZNAVANJE JEZIKA - TRENIRANJE MODELA")
    print("=" * 70)
    print()
    
    # 1. Učitaj konfiguraciju
    print("1. Učitavanje konfiguracije...")
    config = load_config(args.config)
    print(f"   ✓ Konfiguracija učitana iz: {args.config}")
    print()
    
    # 2. Inicijalizuj komponente
    print("2. Inicijalizacija komponenti...")
    
    audio_processor = AudioProcessor(
        target_sr=config['audio']['target_sample_rate']
    )
    print(f"   ✓ AudioProcessor inicijalizovan (target_sr={config['audio']['target_sample_rate']})")
    
    feature_extractor = FeatureExtractor(
        n_mfcc=config['features']['n_mfcc'],
        n_fft=config['features']['n_fft'],
        hop_length=config['features']['hop_length'],
        n_mels=config['features']['n_mels']
    )
    print(f"   ✓ FeatureExtractor inicijalizovan (n_mfcc={config['features']['n_mfcc']})")
    
    dataset_builder = DatasetBuilder(
        data_dir=args.data_dir,
        audio_processor=audio_processor,
        feature_extractor=feature_extractor
    )
    print(f"   ✓ DatasetBuilder inicijalizovan (data_dir={args.data_dir})")
    print()
    
    # 3. Pripremi dataset
    print("3. Priprema dataseta...")
    print("-" * 70)
    
    if args.load_dataset and os.path.exists(args.processed_dir):
        print(f"   Učitavanje postojećeg dataseta iz: {args.processed_dir}")
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = \
            DatasetBuilder.load_saved_dataset(args.processed_dir)
    else:
        print(f"   Izgradnja novog dataseta iz: {args.data_dir}")
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = \
            dataset_builder.build_dataset(
                max_length=config['dataset']['max_sequence_length'],
                feature_type='mfcc'
            )
        
        # Sačuvaj dataset
        if args.save_dataset:
            print(f"\n   Čuvanje dataseta u: {args.processed_dir}")
            dataset_builder.save_dataset(
                X_train, X_val, X_test, y_train, y_val, y_test,
                output_dir=args.processed_dir
            )
    
    print("-" * 70)
    print()
    
    # Konvertuj labele u one-hot encoding
    num_classes = len(label_encoder.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"   Dataset pripremljen:")
    print(f"   - Broj klasa (jezika): {num_classes}")
    print(f"   - Jezici: {', '.join(label_encoder.classes_)}")
    print(f"   - Train: {X_train.shape[0]} uzoraka")
    print(f"   - Validation: {X_val.shape[0]} uzoraka")
    print(f"   - Test: {X_test.shape[0]} uzoraka")
    print()
    
    # Kreiraj models direktorijum
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    
    # 4. Treniraj CNN model
    if args.train_cnn:
        print("4. Treniranje CNN modela...")
        print("=" * 70)
        
        # Pripremi podatke za CNN
        X_train_cnn, X_val_cnn, X_test_cnn = prepare_data_for_cnn(X_train, X_val, X_test)
        
        # Kreiraj i izgradi CNN model
        cnn_input_shape = X_train_cnn.shape[1:]
        cnn_config = config.get('cnn_model', {})
        cnn_model = CNNLanguageClassifier(
            input_shape=cnn_input_shape,
            num_classes=num_classes,
            filters=cnn_config.get('filters', [32, 64, 128]),
            dense_units=cnn_config.get('dense_units', 128),
            dropout_rate=cnn_config.get('dropout_rate', 0.5)
        )
        cnn_model.build_model()
        
        print(f"\n   CNN arhitektura:")
        cnn_model.model.summary()
        print()
        
        # Treniraj model
        print("   Treniranje u toku...")
        cnn_history = cnn_model.train(
            X_train_cnn, y_train_cat,
            X_val_cnn, y_val_cat,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            early_stopping_patience=config['training']['early_stopping_patience']
        )
        
        # Evaluiraj na test skupu
        print("\n   Evaluacija na test skupu:")
        cnn_metrics = cnn_model.evaluate(X_test_cnn, y_test_cat)
        print(f"   - Test Accuracy: {cnn_metrics['accuracy']:.4f}")
        print(f"   - Test Loss: {cnn_metrics['loss']:.4f}")
        
        # Sačuvaj model
        cnn_model_path = os.path.join(args.models_dir, 'cnn_model.h5')
        cnn_model.save_model(cnn_model_path)
        
        # Prikaži training history
        plot_training_history(cnn_history, 'CNN_Model', args.models_dir)
        
        print("=" * 70)
        print()
    
    # 5. Treniraj RNN model
    if args.train_rnn:
        print("5. Treniranje RNN/LSTM modela...")
        print("=" * 70)
        
        # Pripremi podatke za RNN
        X_train_rnn, X_val_rnn, X_test_rnn = prepare_data_for_rnn(X_train, X_val, X_test)
        
        # Kreiraj i izgradi RNN model
        rnn_input_shape = X_train_rnn.shape[1:]
        rnn_config = config.get('rnn_model', {})
        rnn_model = RNNLanguageClassifier(
            input_shape=rnn_input_shape,
            num_classes=num_classes,
            lstm_units=rnn_config.get('lstm_units', 128),
            dropout_rate=rnn_config.get('dropout_rate', 0.3),
            dense_units=rnn_config.get('dense_units', 64)
        )
        rnn_model.build_model()
        
        print(f"\n   RNN/LSTM arhitektura:")
        rnn_model.model.summary()
        print()
        
        # Treniraj model
        print("   Treniranje u toku...")
        rnn_history = rnn_model.train(
            X_train_rnn, y_train_cat,
            X_val_rnn, y_val_cat,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            early_stopping_patience=config['training']['early_stopping_patience']
        )
        
        # Evaluiraj na test skupu
        print("\n   Evaluacija na test skupu:")
        rnn_metrics = rnn_model.evaluate(X_test_rnn, y_test_cat)
        print(f"   - Test Accuracy: {rnn_metrics['accuracy']:.4f}")
        print(f"   - Test Loss: {rnn_metrics['loss']:.4f}")
        
        # Sačuvaj model
        rnn_model_path = os.path.join(args.models_dir, 'rnn_model.h5')
        rnn_model.save_model(rnn_model_path)
        
        # Prikaži training history
        plot_training_history(rnn_history, 'RNN_Model', args.models_dir)
        
        print("=" * 70)
        print()
    
    # 6. Treniraj Wav2Vec model
    if args.train_wav2vec:
        print("6. Treniranje Wav2Vec modela...")
        print("=" * 70)
        
        # Pripremi podatke za Wav2Vec (isti kao RNN)
        X_train_wav2vec, X_val_wav2vec, X_test_wav2vec = prepare_data_for_rnn(X_train, X_val, X_test)
        
        # Kreiraj i izgradi Wav2Vec model
        wav2vec_input_shape = X_train_wav2vec.shape[1:]
        wav2vec_config = config.get('wav2vec_model', {})
        wav2vec_model = Wav2VecLanguageClassifier(
            input_shape=wav2vec_input_shape,
            num_classes=num_classes,
            lstm_units=wav2vec_config.get('lstm_units', 128),
            dropout_rate=wav2vec_config.get('dropout_rate', 0.3),
            dense_units=wav2vec_config.get('dense_units', 64)
        )
        wav2vec_model.build_model()
        
        print(f"\n   Wav2Vec arhitektura:")
        wav2vec_model.model.summary()
        print()
        
        # Treniraj model
        print("   Treniranje u toku...")
        wav2vec_history = wav2vec_model.train(
            X_train_wav2vec, y_train_cat,
            X_val_wav2vec, y_val_cat,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            early_stopping_patience=config['training']['early_stopping_patience']
        )
        
        # Evaluiraj na test skupu
        print("\n   Evaluacija na test skupu:")
        wav2vec_metrics = wav2vec_model.evaluate(X_test_wav2vec, y_test_cat)
        print(f"   - Test Accuracy: {wav2vec_metrics['accuracy']:.4f}")
        print(f"   - Test Loss: {wav2vec_metrics['loss']:.4f}")
        
        # Sačuvaj model
        wav2vec_model_path = os.path.join(args.models_dir, 'wav2vec_model.h5')
        wav2vec_model.save_model(wav2vec_model_path)
        
        # Prikaži training history
        plot_training_history(wav2vec_history, 'Wav2Vec_Model', args.models_dir)
        
        print("=" * 70)
        print()
    
    # 7. Treniraj Hybrid CNN-RNN model
    if args.train_hybrid_cnn_rnn:
        print("7. Treniranje Hybrid CNN-RNN modela...")
        print("=" * 70)
        
        # Pripremi podatke za Hybrid (isti kao CNN)
        X_train_hybrid, X_val_hybrid, X_test_hybrid = prepare_data_for_cnn(X_train, X_val, X_test)
        
        # Kreiraj i izgradi Hybrid model
        hybrid_input_shape = X_train_hybrid.shape[1:]
        hybrid_config = config.get('hybrid_cnn_rnn_model', {})
        hybrid_model = HybridCnnRnnLanguageClassifier(
            input_shape=hybrid_input_shape,
            num_classes=num_classes,
            cnn_filters=hybrid_config.get('cnn_filters', [32, 64]),
            lstm_units=hybrid_config.get('lstm_units', [128, 64]),
            dropout_rate=hybrid_config.get('dropout_rate', 0.3),
            dense_units=hybrid_config.get('dense_units', 64)
        )
        hybrid_model.build_model()
        
        print(f"\n   Hybrid CNN-RNN arhitektura:")
        hybrid_model.model.summary()
        print()
        
        # Treniraj model
        print("   Treniranje u toku...")
        hybrid_history = hybrid_model.train(
            X_train_hybrid, y_train_cat,
            X_val_hybrid, y_val_cat,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            early_stopping_patience=config['training']['early_stopping_patience']
        )
        
        # Evaluiraj na test skupu
        print("\n   Evaluacija na test skupu:")
        hybrid_metrics = hybrid_model.evaluate(X_test_hybrid, y_test_cat)
        print(f"   - Test Accuracy: {hybrid_metrics['accuracy']:.4f}")
        print(f"   - Test Loss: {hybrid_metrics['loss']:.4f}")
        
        # Sačuvaj model
        hybrid_model_path = os.path.join(args.models_dir, 'hybrid_cnn_rnn_model.h5')
        hybrid_model.save_model(hybrid_model_path)
        
        # Prikaži training history
        plot_training_history(hybrid_history, 'Hybrid_CNN_RNN_Model', args.models_dir)
        
        print("=" * 70)
        print()
    
    # 8. Treniraj SVM model (Classic ML)
    if args.train_svm:
        print("8. Treniranje SVM modela (Classic ML)...")
        print("=" * 70)
        
        # SVM koristi MFCC kao RNN (ali ekstraktuje statističke features interno)
        # Koristimo originalne MFCC podatke (pre transpozicije)
        X_train_svm = X_train
        X_val_svm = X_val
        X_test_svm = X_test
        
        # Kreiraj i izgradi SVM model
        svm_config = config.get('svm_model', {})
        svm_model = SVMLanguageClassifier(
            num_classes=num_classes,
            kernel=svm_config.get('kernel', 'rbf'),
            C=svm_config.get('C', 1.0),
            gamma=svm_config.get('gamma', 'auto')
        )
        svm_model.build_model()
        
        print(f"\n   SVM parametri:")
        print(f"   - Kernel: {svm_model.kernel}")
        print(f"   - C: {svm_model.C}")
        print(f"   - Gamma: {svm_model.gamma}")
        print(f"   - Features: Statistical (mean, std, min, max) from MFCC")
        print()
        
        # Treniraj model
        print("   Treniranje u toku...")
        svm_history = svm_model.train(
            X_train_svm, y_train_cat,
            X_val_svm, y_val_cat
        )
        
        # Evaluiraj na test skupu
        print("\n   Evaluacija na test skupu:")
        svm_metrics = svm_model.evaluate(X_test_svm, y_test_cat)
        print(f"   - Test Accuracy: {svm_metrics['accuracy']:.4f}")
        
        # Sačuvaj model
        svm_model_path = os.path.join(args.models_dir, 'svm_model.pkl')
        svm_model.save_model(svm_model_path)
        
        # Prikaži training history (jednostavan za SVM)
        plot_training_history(svm_history, 'SVM_Model', args.models_dir)
        
        print("=" * 70)
        print()
    
    # 9. Sačuvaj label encoder
    import pickle
    label_encoder_path = os.path.join(args.models_dir, 'label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Label encoder sačuvan: {label_encoder_path}")
    print()
    
    print("=" * 70)
    print("TRENIRANJE ZAVRŠENO!")
    print("=" * 70)
    print(f"\nSačuvani fajlovi u direktorijumu '{args.models_dir}':")
    if args.train_cnn:
        print(f"  - cnn_model.h5")
        print(f"  - CNN_Model_training_history.png")
    if args.train_rnn:
        print(f"  - rnn_model.h5")
        print(f"  - RNN_Model_training_history.png")
    if args.train_wav2vec:
        print(f"  - wav2vec_model.h5")
        print(f"  - Wav2Vec_Model_training_history.png")
    if args.train_hybrid_cnn_rnn:
        print(f"  - hybrid_cnn_rnn_model.h5")
        print(f"  - Hybrid_CNN_RNN_Model_training_history.png")
    if args.train_svm:
        print(f"  - svm_model.pkl")
        print(f"  - SVM_Model_training_history.png")
    print(f"  - label_encoder.pkl")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Treniranje CNN i RNN modela za prepoznavanje jezika'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Direktorijum sa audio zapisima organizovanim po jezicima (default: data/raw)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Putanja do konfiguracione datoteke (default: config.yaml)'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Direktorijum za čuvanje treniranih modela (default: models)'
    )
    
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Direktorijum za čuvanje/učitavanje procesovanog dataseta (default: data/processed)'
    )
    
    parser.add_argument(
        '--train-cnn',
        action='store_true',
        default=True,
        help='Treniraj CNN model (default: True)'
    )
    
    parser.add_argument(
        '--train-rnn',
        action='store_true',
        default=True,
        help='Treniraj RNN model (default: True)'
    )
    
    parser.add_argument(
        '--train-wav2vec',
        action='store_true',
        default=True,
        help='Treniraj Wav2Vec model (default: True)'
    )
    
    parser.add_argument(
        '--train-hybrid-cnn-rnn',
        action='store_true',
        default=True,
        help='Treniraj Hybrid CNN-RNN model (default: True)'
    )
    
    parser.add_argument(
        '--train-svm',
        action='store_true',
        default=True,
        help='Treniraj SVM model (Classic ML) (default: True)'
    )
    
    parser.add_argument(
        '--no-cnn',
        action='store_true',
        help='Preskoči treniranje CNN modela'
    )
    
    parser.add_argument(
        '--no-rnn',
        action='store_true',
        help='Preskoči treniranje RNN modela'
    )
    
    parser.add_argument(
        '--no-wav2vec',
        action='store_true',
        help='Preskoči treniranje Wav2Vec modela'
    )
    
    parser.add_argument(
        '--no-hybrid-cnn-rnn',
        action='store_true',
        help='Preskoči treniranje Hybrid CNN-RNN modela'
    )
    
    parser.add_argument(
        '--no-svm',
        action='store_true',
        help='Preskoči treniranje SVM modela'
    )
    
    parser.add_argument(
        '--save-dataset',
        action='store_true',
        default=True,
        help='Sačuvaj procesovani dataset (default: True)'
    )
    
    parser.add_argument(
        '--load-dataset',
        action='store_true',
        help='Učitaj prethodno sačuvan dataset umesto ponovne obrade'
    )
    
    args = parser.parse_args()
    
    # Obradi --no-* flagove
    if args.no_cnn:
        args.train_cnn = False
    if args.no_rnn:
        args.train_rnn = False
    if args.no_wav2vec:
        args.train_wav2vec = False
    if args.no_hybrid_cnn_rnn:
        args.train_hybrid_cnn_rnn = False
    if args.no_svm:
        args.train_svm = False
    
    # Proveri da li je bar jedan model izabran
    if not args.train_cnn and not args.train_rnn and not args.train_wav2vec and not args.train_hybrid_cnn_rnn and not args.train_svm:
        print("Greška: Morate izabrati bar jedan model za treniranje")
        print("Koristite --train-cnn, --train-rnn, --train-wav2vec, --train-hybrid-cnn-rnn i/ili --train-svm")
        exit(1)
    
    main(args)
