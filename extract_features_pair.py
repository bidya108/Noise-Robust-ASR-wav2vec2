import os
import torch
import torchaudio

SAMPLE_RATE = 16000
N_MELS = 80

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    hop_length=160,
    n_mels=N_MELS
)
to_db = torchaudio.transforms.AmplitudeToDB()

def load_audio(path: str):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav

def find_clean_flac(librispeech_root: str, utt_id: str) -> str:
    spk, chap, _ = utt_id.split("-")
    return os.path.join(librispeech_root, spk, chap, utt_id + ".flac")

def extract_from_metadata(metadata_path: str, mode: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    kept = 0
    for i, line in enumerate(lines, 1):
        utt_id, _text = line.split("|", 1)

        if mode == "clean":
            audio_path = find_clean_flac("datasets/librispeech/train-clean-100", utt_id)
        elif mode == "noisy":
            audio_path = os.path.join("datasets/noisy_10k_wavs", utt_id + ".wav")
        else:
            raise ValueError("mode must be 'clean' or 'noisy'")

        if not os.path.exists(audio_path):
            continue

        wav = load_audio(audio_path)

        feats = mel(wav)
        feats = to_db(feats)
        feats = feats.squeeze(0).transpose(0, 1).contiguous() 

        torch.save(feats, os.path.join(out_dir, utt_id + ".pt"))
        kept += 1

        if i % 500 == 0:
            print(f"[{mode}] processed {i}/{len(lines)} | saved {kept}")

    print(f"Done [{mode}] saved features:", kept, "to", out_dir)

if __name__ == "__main__":
    extract_from_metadata(
        metadata_path="datasets/metadata_10k_clean.txt",
        mode="clean",
        out_dir="datasets/features_clean_10k"
    )
    extract_from_metadata(
        metadata_path="datasets/metadata_10k_noisy.txt",
        mode="noisy",
        out_dir="datasets/features_noisy_10k"
    )