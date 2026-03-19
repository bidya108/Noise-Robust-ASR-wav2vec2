import torch
import torchaudio
import numpy as np
from pyctcdecode import build_ctcdecoder
from torch.utils.data import DataLoader
from functools import partial

from src.data.dataset_wave import ASRWaveDataset
from src.utils.metrics import wer, norm_text

def collate_fn(batch):
    wavs, texts = zip(*batch)

    wav_lens = torch.tensor([w.numel() for w in wavs], dtype=torch.long)
    max_len = int(wav_lens.max().item())

    padded = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        padded[i, : w.numel()] = w

    return padded, wav_lens, texts

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)

    model.load_state_dict(torch.load("artifacts/checkpoints/w2v2_best.pt", map_location=device))
    model.eval()

    labels = bundle.get_labels()
    decoder_labels = list(labels)
    decoder_labels[0] = ""
    decoder = build_ctcdecoder(decoder_labels)

    test_ds = ASRWaveDataset("datasets/meta_test_fullpath.txt")
    loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    total_wer = 0.0
    n = 0

    with torch.no_grad():
        for wav, wav_lens, texts in loader:
            wav = wav.to(device)
            wav_lens = wav_lens.to(device)

            out, out_lens = model(wav, wav_lens)
            probs = out.exp().cpu().numpy()
            out_lens = out_lens.cpu().tolist()

            for i in range(len(texts)):
                hyp = decoder.decode(probs[i][:out_lens[i]]).lower().strip()
                ref = texts[i]

                total_wer += wer(norm_text(ref), norm_text(hyp))
                n += 1

    print(f"\n🔥 FINAL TEST WER: {total_wer / n:.6f}")

if __name__ == "__main__":
    main()