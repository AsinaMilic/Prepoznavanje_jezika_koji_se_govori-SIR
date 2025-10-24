#!/usr/bin/env python
"""
Standalone script za brzo prepoznavanje jezika iz audio zapisa.

Ovaj script omogućava jednostavno prepoznavanje jezika iz audio fajla
bez potrebe za korišćenjem CLI interfejsa.

Primer korišćenja:
    python predict.py sample.wav
    python predict.py sample.wav --model models/cnn_model.h5
    python predict.py sample.wav --top-k 5
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

from src.language_recognizer import LanguageRecognizer, LanguageRecognizerError


def format_results(predictions, processing_time: float, audio_path: str):
    """
    Formatira rezultate prepoznavanja u čitljiv format.
    
    Args:
        predictions: Lista (language, probability) tuples
        processing_time: Vreme obrade u sekundama
        audio_path: Putanja do audio fajla
    """
    print("\n" + "=" * 70)
    print("REZULTATI PREPOZNAVANJA JEZIKA")
    print("=" * 70)
    print(f"\nAudio fajl: {audio_path}")
    print(f"Vreme obrade: {processing_time:.3f}s")
    print("\n" + "-" * 70)
    print(f"{'Rang':<8} {'Jezik':<20} {'Verovatnoća':<15} {'Procenat':<10}")
    print("-" * 70)
    
    for idx, (language, probability) in enumerate(predictions, 1):
        bar_length = int(probability * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        print(f"{idx:<8} {language.upper():<20} {probability:.6f}      {probability*100:>6.2f}%")
        print(f"         {bar}")
    
    print("-" * 70)
    
    # Prikaži prepoznati jezik
    best_language, best_prob = predictions[0]
    confidence_level = "VISOKA" if best_prob > 0.8 else "SREDNJA" if best_prob > 0.5 else "NISKA"
    
    print(f"\n✓ PREPOZNATI JEZIK: {best_language.upper()}")
    print(f"  Pouzdanost: {confidence_level} ({best_prob*100:.2f}%)")
    print("=" * 70 + "\n")


def find_model_path(model_type: str = 'cnn') -> Optional[str]:
    """
    Pronalazi putanju do modela u standardnom models direktorijumu.
    
    Args:
        model_type: Tip modela ('cnn' ili 'rnn')
        
    Returns:
        Putanja do modela ili None ako nije pronađen
    """
    models_dir = Path('models')
    model_path = models_dir / f'{model_type}_model.h5'
    
    if model_path.exists():
        return str(model_path)
    
    return None


def find_label_encoder_path() -> Optional[str]:
    """
    Pronalazi putanju do label encoder-a.
    
    Returns:
        Putanja do label encoder-a ili None ako nije pronađen
    """
    label_encoder_path = Path('models') / 'label_encoder.pkl'
    
    if label_encoder_path.exists():
        return str(label_encoder_path)
    
    return None


def main():
    """
    Glavna funkcija za prepoznavanje jezika.
    """
    parser = argparse.ArgumentParser(
        description='Prepoznavanje jezika iz audio zapisa',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Primeri korišćenja:
  # Osnovno prepoznavanje (koristi CNN model)
  python predict.py sample.wav

  # Korišćenje RNN modela
  python predict.py sample.wav --model-type rnn

  # Prikaži top 5 jezika
  python predict.py sample.wav --top-k 5

  # Eksplicitna putanja do modela
  python predict.py sample.wav --model models/cnn_model.h5

Napomena:
  - Podrazumevani model je CNN (models/cnn_model.h5)
  - Label encoder mora biti u models/label_encoder.pkl
  - Podržani formati: WAV, MP3, FLAC
        """
    )
    
    parser.add_argument(
        'audio',
        type=str,
        help='Putanja do audio fajla za prepoznavanje'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Putanja do treniranog modela (default: models/cnn_model.h5)'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['cnn', 'rnn', 'wav2vec', 'hybrid_cnn_rnn', 'svm'],
        default='cnn',
        help='Tip modela ako --model nije specificiran (default: cnn)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Broj top jezika za prikaz (default: 3)'
    )
    
    parser.add_argument(
        '--label-encoder',
        type=str,
        default=None,
        help='Putanja do label encoder-a (default: models/label_encoder.pkl)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Prikaži samo prepoznati jezik bez dodatnih informacija'
    )
    
    args = parser.parse_args()
    
    # Proveri da li audio fajl postoji
    if not os.path.exists(args.audio):
        print(f"Greška: Audio fajl ne postoji: {args.audio}", file=sys.stderr)
        sys.exit(1)
    
    # Odredi putanju do modela
    if args.model:
        model_path = args.model
        # Odredi tip modela iz putanje
        if 'svm' in model_path.lower():
            model_type = 'svm'
        elif 'hybrid_cnn_rnn' in model_path.lower():
            model_type = 'hybrid_cnn_rnn'
        elif 'cnn' in model_path.lower():
            model_type = 'cnn'
        elif 'wav2vec' in model_path.lower():
            model_type = 'wav2vec'
        else:
            model_type = 'rnn'
    else:
        model_path = find_model_path(args.model_type)
        model_type = args.model_type
        
        if not model_path:
            print(f"Greška: Model nije pronađen: models/{args.model_type}_model.h5", file=sys.stderr)
            print(f"\nMolimo vas da prvo trenirate model koristeći:", file=sys.stderr)
            print(f"  python train.py --data-dir data/raw", file=sys.stderr)
            print(f"ili:", file=sys.stderr)
            print(f"  python cli.py train --data-dir data/raw --model-type {args.model_type}", file=sys.stderr)
            sys.exit(1)
    
    # Proveri da li model postoji
    if not os.path.exists(model_path):
        print(f"Greška: Model ne postoji: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Odredi putanju do label encoder-a
    if args.label_encoder:
        label_encoder_path = args.label_encoder
    else:
        label_encoder_path = find_label_encoder_path()
        
        if not label_encoder_path:
            print("Greška: Label encoder nije pronađen: models/label_encoder.pkl", file=sys.stderr)
            print("\nLabel encoder se kreira tokom treniranja modela.", file=sys.stderr)
            sys.exit(1)
    
    # Proveri da li label encoder postoji
    if not os.path.exists(label_encoder_path):
        print(f"Greška: Label encoder ne postoji: {label_encoder_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Učitaj model i inicijalizuj recognizer
        if not args.quiet:
            print(f"\nUčitavanje {model_type.upper()} modela...")
            print(f"Model: {model_path}")
            print(f"Label encoder: {label_encoder_path}")
        
        recognizer = LanguageRecognizer(
            model_path=model_path,
            label_encoder_path=label_encoder_path,
            model_type=model_type
        )
        
        if not args.quiet:
            supported_languages = recognizer.get_supported_languages()
            print(f"Podržani jezici: {', '.join(supported_languages)}")
            print(f"\nPrepoznavanje jezika iz: {args.audio}")
            print("Obrada u toku...")
        
        # Prepoznaj jezik
        predictions = recognizer.recognize(args.audio, top_k=args.top_k)
        processing_time = recognizer.get_last_processing_time()
        
        # Prikaži rezultate
        if args.quiet:
            # Quiet mode - samo prepoznati jezik
            best_language, best_prob = predictions[0]
            print(f"{best_language}")
        else:
            # Normalan mode - detaljni rezultati
            format_results(predictions, processing_time, args.audio)
        
    except LanguageRecognizerError as e:
        print(f"\nGreška pri prepoznavanju jezika: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nPrepoznavanje prekinuto od strane korisnika.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nNeočekivana greška: {e}", file=sys.stderr)
        import traceback
        if not args.quiet:
            print("\nDetalji greške:", file=sys.stderr)
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
