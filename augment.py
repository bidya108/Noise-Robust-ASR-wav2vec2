import os
import random
import torch
import torchaudio

def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2) + 1e-12)

def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav

def _ensure_sr(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav

def _pad_or_crop(noise: torch.Tensor, target_len: int) -> torch.Tensor:
    if noise.size(1) < target_len:
        reps = (target_len // noise.size(1)) + 1
        noise = noise.repeat(1, reps)[:, :target_len]
    else:
        noise = noise[:, :target_len]
    return noise

def mix_with_noise(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    clean_r = _rms(clean)
    noise_r = _rms(noise)
    desired_noise_r = clean_r / (10 ** (snr_db / 20))
    scale = desired_noise_r / (noise_r + 1e-12)
    noisy = clean + noise * scale
    return torch.clamp(noisy, -1.0, 1.0)

class WaveAugmenter:
    def __init__(
        self,
        demand_root: str,
        sample_rate: int = 16000,
        p_noise: float = 0.8,
        snr_min: float = 0.0,
        snr_max: float = 20.0,
        p_gain: float = 0.3,
        gain_min: float = 0.8,
        gain_max: float = 1.2,
        p_shift: float = 0.3,
        shift_max_ms: float = 30.0,
        seed: int = 42
    ):
        random.seed(seed)
        self.sr = sample_rate
        self.p_noise = p_noise
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.p_gain = p_gain
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.p_shift = p_shift
        self.shift_max = int((shift_max_ms / 1000.0) * sample_rate)

        self.noise_files = []
        for root, _, files in os.walk(demand_root):
            for fn in files:
                if fn.lower().endswith((".wav", ".flac")):
                    self.noise_files.append(os.path.join(root, fn))

        if len(self.noise_files) == 0:
            raise RuntimeError(f"No noise files found under: {demand_root}")

    def _load_noise(self) -> torch.Tensor:
        path = random.choice(self.noise_files)
        wav, sr = torchaudio.load(path)
        wav = _to_mono(wav)
        wav = _ensure_sr(wav, sr, self.sr)
        return wav

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if self.p_shift > 0 and random.random() < self.p_shift and self.shift_max > 0:
            shift = random.randint(-self.shift_max, self.shift_max)
            wav = torch.roll(wav, shifts=shift, dims=1)

        if self.p_gain > 0 and random.random() < self.p_gain:
            g = random.uniform(self.gain_min, self.gain_max)
            wav = torch.clamp(wav * g, -1.0, 1.0)

        if self.p_noise > 0 and random.random() < self.p_noise:
            noise = self._load_noise()
            noise = _pad_or_crop(noise, wav.size(1))
            snr = random.uniform(self.snr_min, self.snr_max)
            wav = mix_with_noise(wav, noise, snr)

        return wav