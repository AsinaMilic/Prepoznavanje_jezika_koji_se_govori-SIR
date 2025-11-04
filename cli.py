#!/usr/bin/env python
"""
CLI interfejs za sistem prepoznavanja jezika.
"""

import argparse
import sys
import os
import yaml
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text

from src.audio_processor import AudioProcessor
from src.feature_extractor import FeatureExtractor
from src.dataset_builder import DatasetBuilder
from src.language_recognizer import LanguageRecognizer
from src.evaluator import ModelEvaluator

console = Console()


def load_config(config_path: str = 'config.yaml') -> dict:
    """Učitava konfiguraciju iz YAML fajla."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Greška pri učitavanju konfiguracije: {e}[/red]")
        sys.exit(1)


def train_command(args):
    """Komanda za treniranje modela - poziva train.py."""
    import subprocess
    
    console.print(Panel.fit(
        "[bold cyan]Treniranje Modela za Prepoznavanje Jezika[/bold cyan]",
        border_style="cyan"
    ))
    
    # Pripremi argumente za train.py
    train_args = ['python', 'train.py', '--data-dir', args.data_dir]
    
    # Dodaj opcione argumente
    if args.epochs:
        # Privremeno ažuriraj config.yaml
        console.print("[yellow]Napomena: --epochs opcija nije podržana. Koristi config.yaml[/yellow]")
    
    if args.batch_size:
        console.print("[yellow]Napomena: --batch-size opcija nije podržana. Koristi config.yaml[/yellow]")
    
    # Dodaj model type flagove
    model_type = args.model_type.lower()
    
    if model_type not in ['cnn', 'rnn', 'wav2vec', 'hybrid_cnn_rnn', 'svm']:
        console.print("[red]Greška: model-type mora biti 'cnn', 'rnn', 'wav2vec', 'hybrid_cnn_rnn' ili 'svm'[/red]")
        sys.exit(1)
    
    # Dodaj --no-* flagove za sve modele osim izabranog
    all_models = ['cnn', 'rnn', 'wav2vec', 'hybrid_cnn_rnn', 'svm']
    for model in all_models:
        if model != model_type:
            flag_name = model.replace('_', '-')
            train_args.append(f'--no-{flag_name}')
    
    console.print(f"\n[cyan]Pozivam train.py za {model_type.upper()} model...[/cyan]\n")
    
    try:
        # Pokreni train.py
        result = subprocess.run(train_args, check=True)
        
        if result.returncode == 0:
            console.print(f"\n[bold green]✓ Treniranje uspešno završeno![/bold green]")
        
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Greška tokom treniranja: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Treniranje prekinuto od strane korisnika.[/yellow]")
        sys.exit(130)


def recognize_command(args):
    """Komanda za prepoznavanje jezika iz jednog audio fajla."""
    console.print(Panel.fit(
        "[bold cyan]Prepoznavanje Jezika[/bold cyan]",
        border_style="cyan"
    ))
    
    audio_path = args.audio
    model_path = args.model
    top_k = args.top_k
    
    # Proveri da li fajlovi postoje
    if not os.path.exists(audio_path):
        console.print(f"[red]Greška: Audio fajl ne postoji: {audio_path}[/red]")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        console.print(f"[red]Greška: Model ne postoji: {model_path}[/red]")
        sys.exit(1)
    
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
    
    # Putanja do label encoder-a
    label_encoder_path = Path(model_path).parent / 'label_encoder.pkl'
    if not os.path.exists(label_encoder_path):
        console.print(f"[red]Greška: Label encoder ne postoji: {label_encoder_path}[/red]")
        sys.exit(1)
    
    try:
        console.print(f"\n[cyan]Učitavanje modela...[/cyan]")
        recognizer = LanguageRecognizer(
            model_path=model_path,
            label_encoder_path=str(label_encoder_path),
            model_type=model_type
        )
        
        console.print(f"[cyan]Prepoznavanje jezika iz: {audio_path}[/cyan]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Obrada audio zapisa...", total=None)
            predictions = recognizer.recognize(audio_path, top_k=top_k)
            progress.update(task, completed=True)
        
        processing_time = recognizer.get_last_processing_time()
        
        # Prikaži rezultate
        results_table = Table(title=f"Top {top_k} Jezika", box=box.ROUNDED)
        results_table.add_column("Rang", style="cyan", justify="center")
        results_table.add_column("Jezik", style="yellow")
        results_table.add_column("Verovatnoća", style="green", justify="right")
        results_table.add_column("Procenat", style="magenta", justify="right")
        
        for idx, (language, probability) in enumerate(predictions, 1):
            results_table.add_row(
                str(idx),
                language.upper(),
                f"{probability:.6f}",
                f"{probability*100:.2f}%"
            )
        
        console.print("\n")
        console.print(results_table)
        console.print(f"\n[dim]Vreme obrade: {processing_time:.3f}s[/dim]")
        
        # Prikaži prepoznati jezik
        best_language, best_prob = predictions[0]
        console.print(f"\n[bold green]Prepoznati jezik: {best_language.upper()} ({best_prob*100:.2f}%)[/bold green]")
        
    except Exception as e:
        console.print(f"\n[red]Greška pri prepoznavanju: {e}[/red]")
        sys.exit(1)


def evaluate_command(args):
    """Komanda za evaluaciju modela na test skupu."""
    console.print(Panel.fit(
        "[bold cyan]Evaluacija Modela[/bold cyan]",
        border_style="cyan"
    ))
    
    model_path = args.model
    test_data_dir = args.test_data
    
    # Proveri da li fajlovi postoje
    if not os.path.exists(model_path):
        console.print(f"[red]Greška: Model ne postoji: {model_path}[/red]")
        sys.exit(1)
    
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
    
    # Putanja do label encoder-a
    label_encoder_path = Path(model_path).parent / 'label_encoder.pkl'
    if not os.path.exists(label_encoder_path):
        console.print(f"[red]Greška: Label encoder ne postoji: {label_encoder_path}[/red]")
        sys.exit(1)
    
    try:
        # Učitaj label encoder
        console.print(f"\n[cyan]Učitavanje label encoder-a...[/cyan]")
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Učitaj model
        console.print(f"[cyan]Učitavanje {model_type.upper()} modela...[/cyan]")
        if model_type == 'cnn':
            from src.models.cnn_model import CNNLanguageClassifier
            model = CNNLanguageClassifier(
                input_shape=(128, 100, 1),
                num_classes=len(label_encoder.classes_)
            )
        elif model_type == 'rnn':
            from src.models.rnn_model import RNNLanguageClassifier
            model = RNNLanguageClassifier(
                input_shape=(100, 40),
                num_classes=len(label_encoder.classes_)
            )
        elif model_type == 'wav2vec':
            from src.models.wav2vec_model import Wav2VecLanguageClassifier
            model = Wav2VecLanguageClassifier(
                input_shape=(100, 40),
                num_classes=len(label_encoder.classes_)
            )
        elif model_type == 'hybrid_cnn_rnn':
            from src.models.hybrid_cnn_rnn_model import HybridCnnRnnLanguageClassifier
            model = HybridCnnRnnLanguageClassifier(
                input_shape=(128, 100, 1),
                num_classes=len(label_encoder.classes_)
            )
        else:  # svm
            from src.models.svm_model import SVMLanguageClassifier
            model = SVMLanguageClassifier(
                num_classes=len(label_encoder.classes_)
            )
        
        model.load_model(model_path)
        
        # Učitaj test podatke
        console.print(f"[cyan]Učitavanje test podataka iz: {test_data_dir}[/cyan]")
        
        # Proveri da li postoje sačuvani test podaci
        test_data_path = Path(test_data_dir)
        if (test_data_path / 'X_test.npy').exists():
            X_test = np.load(test_data_path / 'X_test.npy')
            y_test = np.load(test_data_path / 'y_test.npy')
            
            # Konvertuj u one-hot
            from tensorflow.keras.utils import to_categorical
            y_test_cat = to_categorical(y_test, len(label_encoder.classes_))
            
            # Pripremi podatke za model
            if model_type in ['cnn', 'hybrid_cnn_rnn']:
                if len(X_test.shape) == 3:
                    X_test = np.expand_dims(X_test, axis=-1)
            elif model_type in ['rnn', 'wav2vec']:
                if len(X_test.shape) == 3:
                    X_test = np.transpose(X_test, (0, 2, 1))
            # SVM ne treba transpoziciju
        else:
            console.print("[yellow]Sačuvani test podaci nisu pronađeni. Priprema test dataseta...[/yellow]")
            
            # Učitaj konfiguraciju
            config = load_config()
            
            # Inicijalizuj komponente
            audio_processor = AudioProcessor(target_sr=config['audio']['target_sample_rate'])
            feature_extractor = FeatureExtractor.from_config()
            dataset_builder = DatasetBuilder(test_data_dir, audio_processor, feature_extractor)
            
            # Pripremi dataset
            feature_type = 'mel_spectrogram' if model_type == 'cnn' else 'mfcc'
            _, _, X_test, _, _, y_test, _ = dataset_builder.build_dataset(feature_type=feature_type)
            
            # Konvertuj u one-hot
            from tensorflow.keras.utils import to_categorical
            y_test_cat = to_categorical(y_test, len(label_encoder.classes_))
            
            # Pripremi podatke za model
            if model_type in ['cnn', 'hybrid_cnn_rnn']:
                X_test = np.expand_dims(X_test, axis=-1)
            elif model_type in ['rnn', 'wav2vec']:
                X_test = np.transpose(X_test, (0, 2, 1))
            # SVM ne treba transpoziciju
        
        # Evaluacija
        console.print(f"\n[cyan]Evaluacija modela...[/cyan]")
        evaluator = ModelEvaluator(model, label_encoder)
        metrics = evaluator.evaluate_model(X_test, y_test_cat)
        
        # Prikaži globalne metrike
        global_table = Table(title="Globalne Metrike", box=box.ROUNDED)
        global_table.add_column("Metrika", style="cyan")
        global_table.add_column("Vrednost", style="green")
        
        global_table.add_row("Accuracy", f"{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        global_table.add_row("Precision (Weighted)", f"{metrics['precision_weighted']:.4f}")
        global_table.add_row("Recall (Weighted)", f"{metrics['recall_weighted']:.4f}")
        global_table.add_row("F1-Score (Weighted)", f"{metrics['f1_score_weighted']:.4f}")
        global_table.add_row("Broj test uzoraka", str(metrics['num_samples']))
        
        console.print("\n")
        console.print(global_table)
        
        # Prikaži metrike po jezicima
        lang_table = Table(title="Metrike po Jezicima", box=box.ROUNDED)
        lang_table.add_column("Jezik", style="yellow")
        lang_table.add_column("Precision", style="cyan", justify="right")
        lang_table.add_column("Recall", style="green", justify="right")
        lang_table.add_column("F1-Score", style="magenta", justify="right")
        
        for language, lang_metrics in sorted(metrics['per_language'].items()):
            lang_table.add_row(
                language.upper(),
                f"{lang_metrics['precision']:.4f}",
                f"{lang_metrics['recall']:.4f}",
                f"{lang_metrics['f1_score']:.4f}"
            )
        
        console.print("\n")
        console.print(lang_table)
        
        # Sačuvaj izveštaj
        report_path = Path('models') / f'{model_type}_evaluation_report.txt'
        evaluator.save_evaluation_report(metrics, str(report_path), model_name=f"{model_type.upper()} Model")
        
        console.print(f"\n[green]✓ Izveštaj sačuvan: {report_path}[/green]")
        
    except Exception as e:
        console.print(f"\n[red]Greška pri evaluaciji: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


def realtime_command(args):
    """Komanda za pokretanje real-time GUI aplikacije."""
    console.print(Panel.fit(
        "[bold cyan]Real-time Language Recognition[/bold cyan]",
        border_style="cyan"
    ))
    
    import subprocess
    
    # Pripremi argumente za realtime_app.py
    realtime_args = ['python', 'realtime_app.py']
    
    # Dodaj model-size ako je specificiran
    if args.model_size:
        realtime_args.extend(['--model-size', args.model_size])
        console.print(f"\n[cyan]Pokretanje sa Whisper modelom: {args.model_size}[/cyan]\n")
    else:
        console.print(f"\n[cyan]Pokretanje sa default Whisper modelom (base)[/cyan]\n")
    
    try:
        # Pokreni realtime_app.py
        result = subprocess.run(realtime_args, check=True)
        
        if result.returncode == 0:
            console.print(f"\n[bold green]✓ Aplikacija zatvorena[/bold green]")
        
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Greška tokom pokretanja aplikacije: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Aplikacija prekinuta od strane korisnika.[/yellow]")
        sys.exit(130)


def batch_recognize_command(args):
    """Komanda za batch prepoznavanje jezika."""
    console.print(Panel.fit(
        "[bold cyan]Batch Prepoznavanje Jezika[/bold cyan]",
        border_style="cyan"
    ))
    
    audio_dir = args.audio_dir
    model_path = args.model
    top_k = args.top_k if hasattr(args, 'top_k') and args.top_k else 3
    
    # Proveri da li direktorijum postoji
    if not os.path.exists(audio_dir):
        console.print(f"[red]Greška: Direktorijum ne postoji: {audio_dir}[/red]")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        console.print(f"[red]Greška: Model ne postoji: {model_path}[/red]")
        sys.exit(1)
    
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
    
    # Putanja do label encoder-a
    label_encoder_path = Path(model_path).parent / 'label_encoder.pkl'
    if not os.path.exists(label_encoder_path):
        console.print(f"[red]Greška: Label encoder ne postoji: {label_encoder_path}[/red]")
        sys.exit(1)
    
    try:
        # Pronađi sve audio fajlove
        valid_extensions = {'.wav', '.mp3', '.flac'}
        audio_files = []
        
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if Path(file).suffix.lower() in valid_extensions:
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            console.print(f"[yellow]Nisu pronađeni audio fajlovi u direktorijumu: {audio_dir}[/yellow]")
            return
        
        console.print(f"\n[cyan]Pronađeno {len(audio_files)} audio fajlova[/cyan]")
        
        # Učitaj model
        console.print(f"[cyan]Učitavanje modela...[/cyan]")
        recognizer = LanguageRecognizer(
            model_path=model_path,
            label_encoder_path=str(label_encoder_path),
            model_type=model_type
        )
        
        # Batch prepoznavanje sa progress bar-om
        console.print(f"\n[cyan]Prepoznavanje jezika...[/cyan]\n")
        
        results = []
        with tqdm(total=len(audio_files), desc="Obrada", unit="fajl") as pbar:
            for audio_path in audio_files:
                try:
                    predictions = recognizer.recognize(audio_path, top_k=top_k)
                    processing_time = recognizer.get_last_processing_time()
                    
                    results.append({
                        'audio_path': audio_path,
                        'predictions': predictions,
                        'processing_time': processing_time,
                        'success': True,
                        'error': None
                    })
                except Exception as e:
                    results.append({
                        'audio_path': audio_path,
                        'predictions': [],
                        'processing_time': 0.0,
                        'success': False,
                        'error': str(e)
                    })
                
                pbar.update(1)
        
        # Prikaži rezultate
        console.print("\n")
        results_table = Table(title="Rezultati Batch Prepoznavanja", box=box.ROUNDED)
        results_table.add_column("Fajl", style="cyan", no_wrap=False)
        results_table.add_column("Jezik", style="yellow")
        results_table.add_column("Verovatnoća", style="green", justify="right")
        results_table.add_column("Vreme", style="magenta", justify="right")
        results_table.add_column("Status", style="white", justify="center")
        
        successful = 0
        failed = 0
        total_time = 0.0
        
        for result in results:
            filename = Path(result['audio_path']).name
            
            if result['success']:
                language, probability = result['predictions'][0]
                results_table.add_row(
                    filename,
                    language.upper(),
                    f"{probability*100:.2f}%",
                    f"{result['processing_time']:.2f}s",
                    "[green]✓[/green]"
                )
                successful += 1
                total_time += result['processing_time']
            else:
                results_table.add_row(
                    filename,
                    "-",
                    "-",
                    "-",
                    "[red]✗[/red]"
                )
                failed += 1
        
        console.print(results_table)
        
        # Prikaži statistiku
        stats_table = Table(title="Statistika", box=box.ROUNDED)
        stats_table.add_column("Metrika", style="cyan")
        stats_table.add_column("Vrednost", style="green")
        
        stats_table.add_row("Ukupno fajlova", str(len(audio_files)))
        stats_table.add_row("Uspešno", f"[green]{successful}[/green]")
        stats_table.add_row("Neuspešno", f"[red]{failed}[/red]" if failed > 0 else "0")
        stats_table.add_row("Ukupno vreme", f"{total_time:.2f}s")
        if successful > 0:
            stats_table.add_row("Prosečno vreme", f"{total_time/successful:.2f}s")
        
        console.print("\n")
        console.print(stats_table)
        
        # Prikaži greške ako ih ima
        if failed > 0:
            console.print("\n[yellow]Greške:[/yellow]")
            for result in results:
                if not result['success']:
                    console.print(f"  • {Path(result['audio_path']).name}: {result['error']}")
        
    except Exception as e:
        console.print(f"\n[red]Greška pri batch prepoznavanju: {e}[/red]")
        sys.exit(1)


def main():
    """Glavni entry point za CLI."""
    parser = argparse.ArgumentParser(
        description='Sistem za prepoznavanje jezika iz audio zapisa',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Primeri korišćenja:
  # Treniranje CNN modela
  python cli.py train --data-dir ./data/raw --model-type cnn --epochs 50

  # Prepoznavanje jezika iz jednog fajla
  python cli.py recognize --audio sample.wav --model models/cnn_model.h5

  # Evaluacija modela
  python cli.py evaluate --model models/cnn_model.h5 --test-data data/processed

  # Batch prepoznavanje
  python cli.py batch-recognize --audio-dir ./samples --model models/rnn_model.h5

  # Real-time prepoznavanje sa GUI
  python cli.py realtime
  python cli.py realtime --model-size small
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Dostupne komande')
    
    # Train komanda
    train_parser = subparsers.add_parser('train', help='Treniranje modela')
    train_parser.add_argument('--data-dir', required=True, help='Direktorijum sa audio zapisima')
    train_parser.add_argument('--model-type', required=True, choices=['cnn', 'rnn', 'wav2vec', 'hybrid_cnn_rnn', 'svm'], 
                             help='Tip modela (cnn, rnn, wav2vec, hybrid_cnn_rnn ili svm)')
    train_parser.add_argument('--epochs', type=int, help='Broj epoha (default: iz config.yaml)')
    train_parser.add_argument('--batch-size', type=int, help='Batch size (default: iz config.yaml)')
    
    # Recognize komanda
    recognize_parser = subparsers.add_parser('recognize', help='Prepoznavanje jezika iz audio fajla')
    recognize_parser.add_argument('--audio', required=True, help='Putanja do audio fajla')
    recognize_parser.add_argument('--model', required=True, help='Putanja do treniranog modela')
    recognize_parser.add_argument('--top-k', type=int, default=3, help='Broj top jezika (default: 3)')
    
    # Evaluate komanda
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluacija modela na test skupu')
    evaluate_parser.add_argument('--model', required=True, help='Putanja do treniranog modela')
    evaluate_parser.add_argument('--test-data', required=True, 
                                help='Direktorijum sa test podacima ili sačuvanim .npy fajlovima')
    
    # Batch-recognize komanda
    batch_parser = subparsers.add_parser('batch-recognize', help='Batch prepoznavanje jezika')
    batch_parser.add_argument('--audio-dir', required=True, help='Direktorijum sa audio fajlovima')
    batch_parser.add_argument('--model', required=True, help='Putanja do treniranog modela')
    batch_parser.add_argument('--top-k', type=int, default=3, help='Broj top jezika (default: 3)')
    
    # Realtime komanda
    realtime_parser = subparsers.add_parser('realtime', help='Pokreni real-time GUI za prepoznavanje jezika')
    realtime_parser.add_argument(
        '--model-size',
        type=str,
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Veličina Whisper modela (default: base)'
    )
    
    # Parse argumenti
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Pozovi odgovarajuću komandu
    if args.command == 'train':
        train_command(args)
    elif args.command == 'recognize':
        recognize_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'batch-recognize':
        batch_recognize_command(args)
    elif args.command == 'realtime':
        realtime_command(args)


if __name__ == '__main__':
    main()
