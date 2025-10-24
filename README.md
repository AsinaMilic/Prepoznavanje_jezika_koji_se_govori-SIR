# Sistem za Prepoznavanje Jezika iz Govora

Projekat za automatsko prepoznavanje jezika na osnovu zvučnog signala korišćenjem tehnika dubokog učenja i obrade prirodnog jezika.

## Opis

Sistem prepoznaje jezik koji se govori u audio zapisu bez potrebe za razumevanjem sadržaja. Implementirano je pet različitih pristupa:

- **CNN** - Konvolucione mreže za analizu mel-spektrograma
- **RNN/LSTM** - Rekurentne mreže za sekvencionalne MFCC karakteristike
- **Wav2Vec** - Model inspirisan Wav2Vec arhitekturom
- **Hybrid CNN-RNN** - Kombinovani pristup
- **SVM** - Klasični ML pristup sa statističkim karakteristikama

## Podržani Jezici

- Engleski
- Srpski
- Nemački
- Španski
- Francuski

## Instalacija

Potreban je Python 3.8 ili noviji.

```bash
pip install -r requirements.txt
```

## Struktura Projekta

```
├── src/
│   ├── audio_processor.py       # Učitavanje i preprocesiranje audio zapisa
│   ├── feature_extractor.py     # Ekstrakcija MFCC i mel-spektrograma
│   ├── dataset_builder.py       # Priprema dataseta za treniranje
│   ├── language_recognizer.py   # Glavni interfejs za prepoznavanje
│   ├── evaluator.py            # Evaluacija modela
│   └── models/                 # Implementacije modela
│       ├── cnn_model.py
│       ├── rnn_model.py
│       ├── wav2vec_model.py
│       ├── hybrid_cnn_rnn_model.py
│       └── svm_model.py
├── data/
│   ├── raw/                    # Originalni audio zapisi (po jezicima)
│   └── processed/              # Procesovani podaci
├── models/                     # Trenirani modeli
├── train.py                    # Script za treniranje modela
├── predict.py                  # Script za prepoznavanje jezika
├── cli.py                      # CLI interfejs
└── config.yaml                 # Konfiguracija sistema
```

## Treniranje Modela

Organizuj audio zapise u `data/raw/` direktorijumu po jezicima:

```
data/raw/
├── english/
│   ├── sample1.wav
│   └── sample2.wav
├── serbian/
│   ├── sample1.wav
│   └── sample2.wav
└── ...
```

Treniranje svih modela:

```bash
python train.py --data-dir data/raw
```

Treniranje specifičnog modela:

```bash
# CNN model
python cli.py train --data-dir data/raw --model-type cnn

# RNN model
python cli.py train --data-dir data/raw --model-type rnn

# Hybrid CNN-RNN model
python cli.py train --data-dir data/raw --model-type hybrid_cnn_rnn

# SVM model
python cli.py train --data-dir data/raw --model-type svm
```

## Prepoznavanje Jezika

Osnovno korišćenje:

```bash
python predict.py sample.wav

# Sa specifičnim modelom
python predict.py sample.wav --model models/cnn_model.h5
```

CLI komande:

```bash
# Prepoznavanje jednog fajla
python cli.py recognize --audio sample.wav --model models/cnn_model.h5

# Batch prepoznavanje
python cli.py batch-recognize --audio-dir ./samples --model models/rnn_model.h5

# Evaluacija modela
python cli.py evaluate --model models/cnn_model.h5 --test-data data/processed
```

## Konfiguracija

Parametri sistema se podešavaju u `config.yaml`:

```yaml
audio:
  target_sample_rate: 16000
  max_duration: 10

features:
  n_mfcc: 40
  n_fft: 2048
  hop_length: 512
  n_mels: 128

training:
  epochs: 30
  batch_size: 64
  learning_rate: 0.001
```

## Ekstrakcija Karakteristika

Sistem koristi:

- **MFCC** - Mel-Frequency Cepstral Coefficients
- **Mel-Spektrogram** - Reprezentacija spektra frekvencija na mel skali
- **Spektralne karakteristike** - Centroid, rolloff, zero-crossing rate

## Arhitekture Modela

**CNN Model**
- Konvolucioni slojevi za ekstrakciju prostornih karakteristika iz mel-spektrograma
- Pooling slojevi za redukciju dimenzionalnosti

**RNN/LSTM Model**
- LSTM slojevi za modelovanje temporalnih zavisnosti u MFCC sekvencama
- Dropout za regularizaciju

**Hybrid CNN-RNN Model**
- CNN slojevi za ekstrakciju lokalnih karakteristika
- LSTM slojevi za modelovanje temporalnih odnosa

**SVM Model**
- Ekstrakcija statističkih karakteristika (mean, std, min, max) iz MFCC
- RBF kernel za nelinearnu klasifikaciju

## Performanse

Trenirani modeli postižu sledeće rezultate (na test skupu):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CNN | ~85-90% | ~0.87 | ~0.86 | ~0.86 |
| RNN | ~80-85% | ~0.82 | ~0.81 | ~0.81 |
| Hybrid | ~87-92% | ~0.89 | ~0.88 | ~0.88 |
| SVM | ~75-80% | ~0.77 | ~0.76 | ~0.76 |

Performanse zavise od kvaliteta i količine trening podataka.

## Tehnologije

- Python 3.8+
- TensorFlow/Keras
- librosa
- scikit-learn
- NumPy
- Matplotlib/Seaborn

## Primeri Korišćenja

Python API:

```python
from src.language_recognizer import LanguageRecognizer

# Inicijalizacija
recognizer = LanguageRecognizer(
    model_path='models/cnn_model.h5',
    label_encoder_path='models/label_encoder.pkl',
    model_type='cnn'
)

# Prepoznavanje jezika
predictions = recognizer.recognize('sample.wav', top_k=3)

for language, probability in predictions:
    print(f"{language}: {probability*100:.2f}%")
```

Batch processing:

```python
audio_files = ['sample1.wav', 'sample2.wav', 'sample3.wav']
results = recognizer.batch_recognize(audio_files, top_k=3)

for result in results:
    if result['success']:
        print(f"{result['audio_path']}: {result['predictions'][0][0]}")
```

## Mogućnosti Primene

- Automatska kategorizacija audio sadržaja
- Višejezični call centri
- Analiza medijskog sadržaja
- Edukativne aplikacije
- Istraživanje u oblasti fonetike i lingvistike

## Autor

Aleksa Milić - SIR 1610