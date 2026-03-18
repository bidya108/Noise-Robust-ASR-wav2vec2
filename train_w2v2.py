import os
import math
import warnings
from functools import partial

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder

from src.data.dataset_wave import ASRWaveDataset
from src.utils.metrics import wer, norm_text

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def build_labels_and_maps(labels):
    label2id = {c: i for i, c in enumerate(labels)}
    id2label = {i: c for i, c in enumerate(labels)}
    return label2id, id2label


def text_to_targets(text, label2id):
    t = text.upper().replace(" ", "|")
    out = []
    for ch in t:
        if ch in label2id:
            out.append(label2id[ch])
    return out


def collate_w2v2(batch, label2id, max_seconds=22, sample_rate=16000):
    wavs, texts = zip(*batch)

    max_samples = max_seconds * sample_rate

    trimmed_wavs = []
    trimmed_texts = []

    for w, t in zip(wavs, texts):
        if w.numel() > max_samples:
            w = w[:max_samples]
        trimmed_wavs.append(w)
        trimmed_texts.append(t)

    wavs = trimmed_wavs
    texts = trimmed_texts

    wav_lens = torch.tensor([w.numel() for w in wavs], dtype=torch.long)
    max_len = int(wav_lens.max().item())

    padded = torch.zeros(len(wavs), max_len, dtype=torch.float32)
    for i, w in enumerate(wavs):
        padded[i, : w.numel()] = w

    targets_list = [torch.tensor(text_to_targets(t, label2id), dtype=torch.long) for t in texts]
    tgt_lens = torch.tensor([t.numel() for t in targets_list], dtype=torch.long)
    targets = torch.cat(targets_list, dim=0) if len(targets_list) > 0 else torch.tensor([], dtype=torch.long)

    return padded, wav_lens, targets, tgt_lens, list(texts)


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=1e-6):
        self.opt = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.step_num = 0

    def step(self):
        self.step_num += 1

        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * (self.step_num / self.warmup_steps)
        else:
            t = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            t = min(max(t, 0.0), 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))

        for pg in self.opt.param_groups:
            pg["lr"] = lr

        return lr


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    train_meta = "datasets/meta_train_fullpath.txt"
    dev_meta = "datasets/meta_dev_fullpath.txt"

    if not os.path.exists(train_meta):
        raise FileNotFoundError(f"Missing train metadata file: {train_meta}")
    if not os.path.exists(dev_meta):
        raise FileNotFoundError(f"Missing dev metadata file: {dev_meta}")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)

    labels = bundle.get_labels()
    print("Labels:", labels)

    label2id, id2label = build_labels_and_maps(labels)

    blank_id = 0
    print("Blank ID:", blank_id)

    decoder_labels = list(labels)
    decoder_labels[0] = ""
    decoder = build_ctcdecoder(decoder_labels)

    train_ds = ASRWaveDataset(metadata_path=train_meta)
    val_ds = ASRWaveDataset(metadata_path=dev_meta)

    print("Train:", len(train_ds))
    print("Val:", len(val_ds))

    num_workers = 2
    batch_size = 3
    max_seconds = 22

    collate_fn = partial(
        collate_w2v2,
        label2id=label2id,
        max_seconds=max_seconds,
        sample_rate=16000,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
        pin_memory=False,
    )

    # Freeze only feature extractor initially
    if hasattr(model, "feature_extractor"):
        for p in model.feature_extractor.parameters():
            p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameter tensors:", len(trainable_params))

    ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    epochs = 25
    base_lr = 2e-5
    weight_decay = 1e-5
    accum_steps = 6

    total_steps = epochs * math.ceil(len(train_loader) / accum_steps)
    warmup_steps = 300

    opt = torch.optim.AdamW(
        trainable_params,
        lr=base_lr,
        weight_decay=weight_decay,
    )
    sched = WarmupCosine(opt, warmup_steps, total_steps, base_lr)

    os.makedirs("artifacts/checkpoints", exist_ok=True)

    best_wer = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for bi, (wav, wav_lens, targets, tgt_lens, _) in enumerate(train_loader):
            wav = wav.to(device)
            wav_lens = wav_lens.to(device)

            out, out_lens = model(wav, wav_lens)
            out = out.log_softmax(dim=-1)
            out_t = out.transpose(0, 1)

            raw_loss = ctc(
                out_t.float().cpu(),
                targets.cpu(),
                out_lens.cpu(),
                tgt_lens.cpu(),
            )

            total_loss += float(raw_loss.item())

            loss = raw_loss / accum_steps
            loss.backward()

            should_step = ((bi + 1) % accum_steps == 0) or ((bi + 1) == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                lr = sched.step()
                opt.zero_grad(set_to_none=True)
            else:
                lr = opt.param_groups[0]["lr"]

            if bi % 50 == 0:
                print(f"Epoch {ep} | Batch {bi} | Loss {raw_loss.item():.4f} | LR {lr:.3e}")

        avg_train_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {ep} Train Loss: {avg_train_loss:.6f}")

        torch.save(model.state_dict(), "artifacts/checkpoints/w2v2_last.pt")

        model.eval()
        total_wer = 0.0
        n = 0

        with torch.no_grad():
            for wav, wav_lens, targets, tgt_lens, texts in val_loader:
                wav = wav.to(device)
                wav_lens = wav_lens.to(device)

                out, out_lens = model(wav, wav_lens)
                out = out.log_softmax(dim=-1)

                probs = out.exp().cpu().numpy()
                out_lens_list = out_lens.cpu().tolist()

                hyps = []
                for i in range(probs.shape[0]):
                    decoded = decoder.decode(probs[i][: int(out_lens_list[i])])
                    hyps.append(decoded.lower().strip())

                for ref, hyp in zip(texts, hyps):
                    if n < 8:
                        print("REF:", norm_text(ref))
                        print("HYP:", norm_text(hyp))
                        print("---")
                    total_wer += wer(norm_text(ref), norm_text(hyp))
                    n += 1

        val_wer = total_wer / max(1, n)
        print(f"Epoch {ep} WER: {val_wer:.6f}")

        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(model.state_dict(), "artifacts/checkpoints/w2v2_best.pt")
            print("Saved best checkpoint")

    print("Training complete")
    print(f"Best WER: {best_wer:.6f}")


if __name__ == "__main__":
    main()