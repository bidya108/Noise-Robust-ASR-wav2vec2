# Noise-Robust Automatic Speech Recognition using Wav2Vec2-CTC

This project presents a noise-robust Automatic Speech Recognition (ASR) pipeline built using Wav2Vec2-CTC and PyTorch. The goal is to improve speech transcription performance under noisy real-world conditions.

## Project Overview
The system is designed to:
- preprocess and organize speech/audio datasets
- train a Wav2Vec2-CTC based ASR model
- evaluate transcription quality using metrics such as Word Error Rate (WER)
- perform inference on unseen audio samples
- handle noisy speech more effectively than basic ASR pipelines

## Features
- Wav2Vec2-CTC based speech recognition pipeline
- Noise-robust ASR training workflow
- Audio preprocessing and dataset preparation
- Model training and validation
- Inference script for transcription
- Evaluation using WER

## Tech Stack
- Python
- PyTorch
- Torchaudio
- Hugging Face Transformers
- Wav2Vec2
- CTC Loss

## Project Structure
```bash
.
├── datasets/
├── src/
├── train_w2v2.py
├── inference.py
├── requirements.txt
└── README.md
