# 🎙️ Noise-Robust Automatic Speech Recognition (ASR) using Wav2Vec2 + CTC

## 🚀 Overview

This project implements an end-to-end **Automatic Speech Recognition (ASR)** system using a pretrained **Wav2Vec2** model fine-tuned with **Connectionist Temporal Classification (CTC) loss**.

The system processes raw speech audio (`.flac`), performs transcription using **beam search decoding**, and is evaluated on standard **LibriSpeech splits** with proper train/dev/test separation.

👉 The final model achieves **4.15% Word Error Rate (WER) on the test set**, demonstrating strong transcription accuracy and generalization.

---

## 🎯 Key Features

* 🎧 End-to-end ASR from raw audio
* 🧠 Pretrained **Wav2Vec2 (Facebook AI)** backbone
* 🔤 CTC-based sequence modeling
* 🔍 Beam search decoding using `pyctcdecode`
* 🗂️ Proper **train / dev / test split**
* ⚡ Efficient batching, padding, and gradient accumulation
* 🧹 Transcript normalization pipeline

---

## 📊 Results

| Dataset | Samples |        WER |
| ------- | ------: | ---------: |
| Train   |  28,538 |          — |
| Dev     |   2,703 |     0.0421 |
| Test    |   2,620 | **0.0415** |

### 🏆 Best Performance

* **Validation WER:** 4.21%
* **Final Test WER:** **4.15%**

---

## 🗂️ Dataset

* **LibriSpeech Dataset**
* Train: 28,538 samples
* Dev: 2,703 samples
* Test: 2,620 samples

## Dataset Setup

The dataset is not included in this repository due to size limitations.

To run the project:

1. Download LibriSpeech dataset:
   https://www.openslr.org/12/

2. Place it in:
   datasets/librispeech/

3. Generate metadata files using provided scripts.
### Format

```
audio_path.flac | transcript
```

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Torchaudio
* Wav2Vec2
* pyctcdecode
* NumPy

---

## 🏗️ Project Structure

```
noise_robust_asr/
│
├── datasets/
│   ├── librispeech/
│   ├── meta_train_fullpath.txt
│   ├── meta_dev_fullpath.txt
│   ├── meta_test_fullpath.txt
│
├── src/
│   ├── data/
│   │   └── dataset_wave.py
│   ├── utils/
│   │   ├── metrics.py
│   │   └── text.py
│
├── artifacts/
│   └── checkpoints/
│       ├── w2v2_best.pt
│       └── w2v2_last.pt
│
├── train_w2v2.py
├── test_w2v2.py
└── README.md
```

---

## 🔥 Training Details

* Model: `WAV2VEC2_ASR_BASE_960H`
* Loss: CTC Loss
* Optimizer: AdamW
* Learning Rate: 2e-5
* Scheduler: Warmup + Cosine Decay
* Epochs: 25 (early stopping based on WER)
* Batch Size: 3
* Gradient Accumulation: 6
* Max Audio Length: 22 seconds

---

## 🧠 Model Pipeline

1. Load raw audio (`.flac`)
2. Convert to mono + resample (16kHz)
3. Pass through Wav2Vec2 encoder
4. Apply CTC loss
5. Decode predictions using beam search

---

## 📈 Sample Predictions

```
REF: the streets were narrow and unpaved but very fairly clean
HYP: the streets were narrow and unpaved but very fairly clean
```

```
REF: each feature was finished eyelids eyelashes and ears being almost invariably perfect
HYP: each feature was finished eyelids eyelashes and ears being almost invariably perfect
```

```
REF: even in middle age they were still comely and the old grey haired women at their cottage doors had a dignity not to say majesty of their own
HYP: even in middle age they were still comely and the old great haired women at their cottage doors had a dignity not to say majesty of their own
```

---

## 🧪 How to Run

### 1️⃣ Install dependencies

```bash
pip install torch torchaudio pyctcdecode numpy
```

### 2️⃣ Train the model

```bash
python train_w2v2.py
```

### 3️⃣ Evaluate on test set

```bash
python test_w2v2.py
```

---

## 📌 Future Improvements

* Add external language model (KenLM) for better decoding
* Real-time ASR interface (Flask / Streamlit)
* Noise augmentation for robustness
* Speaker adaptation
* Model deployment (API or web app)

