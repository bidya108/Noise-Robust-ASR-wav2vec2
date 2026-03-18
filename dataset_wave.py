import os
import torch
import torchaudio
from torch.utils.data import Dataset

from src.utils.text import normalize_transcript

SAMPLE_RATE = 16000


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def _load_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = _to_mono(wav)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav  # (1, T)


class ASRWaveDataset(Dataset):
    def __init__(self, metadata_path: str, augmenter=None, max_items=None):
        self.items = []
        self.augmenter = augmenter

        missing = 0
        bad_lines = 0

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if "|" not in line:
                    bad_lines += 1
                    continue

                audio_path, text = line.split("|", 1)
                audio_path = audio_path.strip()
                text = normalize_transcript(text)

                if not os.path.exists(audio_path):
                    missing += 1
                    continue

                if len(text) > 0:
                    self.items.append((audio_path, text))

        if max_items is not None:
            self.items = self.items[:max_items]

        print(
            f"[Dataset] loaded={len(self.items)} | missing_audio={missing} | bad_lines={bad_lines} "
            f"from {metadata_path}"
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_path, text = self.items[idx]
        wav = _load_audio(audio_path)

        if self.augmenter is not None:
            wav = self.augmenter(wav)

        wav = wav.squeeze(0).contiguous()
        return wav, text