import os
import random
import torch
import torchaudio

SAMPLE_RATE = 16000
SEED = 42
random.seed(SEED)

CLEAN_ROOT = "datasets/librispeech/train-clean-100"
DEMAND_ROOT = "datasets/demand"

META_IN = "datasets/metadata_10k_clean.txt"

NOISY_WAV_DIR = "datasets/noisy_10k_wavs"
META_OUT = "datasets/metadata_10k_noisy.txt"

SNR_MIN, SNR_MAX = 0, 20 


def find_clean_audio(utt_id: str) -> str:
    spk, chap, _ = utt_id.split("-")
    return os.path.join(CLEAN_ROOT, spk, chap, utt_id + ".flac")


def list_noise_files(demand_root: str):
    noise_files = []
    for root, _, files in os.walk(demand_root):
        for fn in files:
            if fn.lower().endswith((".wav", ".flac")):
                noise_files.append(os.path.join(root, fn))
    if not noise_files:
        raise RuntimeError("No DEMAND noise files found in datasets/demand")
    return noise_files


def load_audio(path: str):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2) + 1e-12)


def pad_or_crop(noise: torch.Tensor, target_len: int) -> torch.Tensor:
    if noise.size(1) < target_len:
        reps = (target_len // noise.size(1)) + 1
        noise = noise.repeat(1, reps)[:, :target_len]
    else:
        noise = noise[:, :target_len]
    return noise


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    noise = pad_or_crop(noise, clean.size(1))
    clean_r = rms(clean)
    noise_r = rms(noise)

    desired_noise_r = clean_r / (10 ** (snr_db / 20))
    scale = desired_noise_r / (noise_r + 1e-12)

    noisy = clean + noise * scale
    return torch.clamp(noisy, -1.0, 1.0)


if __name__ == "__main__":
    os.makedirs(NOISY_WAV_DIR, exist_ok=True)
    noise_files = list_noise_files(DEMAND_ROOT)
    print("Found noise files:", len(noise_files))

    out_lines = []

    with open(META_IN, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for i, line in enumerate(lines, 1):
        utt_id, text = line.split("|", 1)

        clean_path = find_clean_audio(utt_id)
        if not os.path.exists(clean_path):
            continue

        clean = load_audio(clean_path)
        noise_path = random.choice(noise_files)
        noise = load_audio(noise_path)

        snr = random.uniform(SNR_MIN, SNR_MAX)
        noisy = mix_at_snr(clean, noise, snr)

        out_wav_path = os.path.join(NOISY_WAV_DIR, utt_id + ".wav")
        torchaudio.save(out_wav_path, noisy, SAMPLE_RATE)

        out_lines.append(f"{utt_id}|{text}")

        if i % 500 == 0:
            print("Processed", i)

    with open(META_OUT, "w", encoding="utf-8") as f:
        for ln in out_lines:
            f.write(ln + "\n")

    print("Saved noisy wavs to:", NOISY_WAV_DIR)
    print("Wrote:", META_OUT, "count:", len(out_lines))