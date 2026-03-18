import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from functools import partial

from src.data.specaug import spec_augment
from src.data.tokenizer import SimpleTokenizer
from src.data.collate import collate_fn
from src.data.feat_ds import FeatASRDataset
from src.models.transformer_asr import TransformerASR
from src.utils.metrics import wer, norm_text


def collate_with_tokenizer(batch, tokenizer):
    return collate_fn(batch, tokenizer)


def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    # True = PAD
    seq_range = torch.arange(max_len, device=lengths.device)
    return seq_range.unsqueeze(0) >= lengths.unsqueeze(1)


def conv_subsample_len(L: torch.Tensor) -> torch.Tensor:
    # two stride-2 conv layers
    L1 = (L + 1) // 2
    L2 = (L1 + 1) // 2
    return torch.clamp(L2, min=1)


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
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * t))

        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


def main():
    device = torch.device("cpu")
    print("Using device:", device)

    tokenizer = SimpleTokenizer()

    ds = FeatASRDataset(
        meta_path="datasets/meta_20k.txt",
        feat_clean_dir="datasets/features_clean_10k",
        feat_noisy_dir="datasets/features_noisy_10k",
    )
    print("Total samples loaded:", len(ds))

    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print("Train:", len(train_ds), "Val:", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=partial(collate_with_tokenizer, tokenizer=tokenizer),
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=partial(collate_with_tokenizer, tokenizer=tokenizer),
        num_workers=0,
        pin_memory=False
    )

    vocab_size = len(tokenizer.char2idx)

    model = TransformerASR(
        input_dim=80,
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_layers=6,
        dim_ff=1024,
        dropout=0.1,
    ).to(device)

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    base_lr = 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-2)

    epochs = 30
    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = 800
    lr_sched = WarmupCosine(opt, warmup_steps, total_steps, base_lr, min_lr=1e-5)

    os.makedirs("artifacts/checkpoints", exist_ok=True)
    best_wer = 1e9
    patience = 5
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        # ================= TRAIN =================
        model.train()
        total_loss = 0.0

        for bi, batch in enumerate(train_loader):
            feats, targets, in_lens, tgt_lens, _texts = batch
            feats = feats.to(device)
            targets = targets.to(device)
            in_lens = in_lens.to("cpu")
            tgt_lens = tgt_lens.to("cpu")

            if ep >= 4:
                feats = spec_augment(feats, t_max=20, t_masks=1, f_max=10, f_masks=1)

            opt.zero_grad()

            out = model(feats)                      # (B, T', V)
            Tprime = out.size(1)
            out_lens = conv_subsample_len(in_lens)
            out_lens = torch.clamp(out_lens, 1, Tprime)

            pad_mask = make_pad_mask(out_lens, Tprime).to(device)
            out = model(feats, src_key_padding_mask=pad_mask)

            out_log = torch.log_softmax(out, dim=2)  # (B, T', V)
            out_t = out_log.permute(1, 0, 2).contiguous()  # (T', B, V)

            loss = ctc(out_t, targets, out_lens, tgt_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lr = lr_sched.step()

            total_loss += float(loss.item())
            if bi % 50 == 0:
                print(f"Epoch {ep} | Batch {bi} | Loss {loss.item():.4f} | LR {lr:.6f}")

        print(f"Epoch {ep} Avg Train Loss: {total_loss / max(1, len(train_loader)):.4f}")

        # ================= VAL =================
        model.eval()
        total_val_loss = 0.0
        total_wer = 0.0
        n = 0

        with torch.no_grad():
            for batch in val_loader:
                feats, targets, in_lens, tgt_lens, texts = batch
                feats = feats.to(device)
                targets = targets.to(device)
                in_lens = in_lens.to("cpu")
                tgt_lens = tgt_lens.to("cpu")

                out = model(feats)
                Tprime = out.size(1)
                out_lens = conv_subsample_len(in_lens)
                out_lens = torch.clamp(out_lens, 1, Tprime)

                pad_mask = make_pad_mask(out_lens, Tprime).to(device)
                out = model(feats, src_key_padding_mask=pad_mask)

                out_log = torch.log_softmax(out, dim=2)
                out_t = out_log.permute(1, 0, 2).contiguous()
                vloss = ctc(out_t, targets, out_lens, tgt_lens)
                total_val_loss += float(vloss.item())

                pred = torch.argmax(out_log, dim=2)
                decoded_texts = []
                for i, L in enumerate(out_lens):
                    seq = pred[i, : int(L.item())].tolist()
                    decoded_texts.append(tokenizer.decode_ctc_greedy(seq))

                for ref, hyp in zip(texts, decoded_texts):
                    total_wer += wer(norm_text(ref), norm_text(hyp))
                    n += 1

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_wer = total_wer / max(1, n)

        print(f"Epoch {ep} Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch {ep} Val WER : {val_wer:.4f}")
        print("=" * 60)

        torch.save(model.state_dict(), "artifacts/checkpoints/last.pt")

        if val_wer < best_wer:
            best_wer = val_wer
            bad_epochs = 0
            torch.save(model.state_dict(), "artifacts/checkpoints/best.pt")
            print("✅ saved best checkpoint")
        else:
            bad_epochs += 1
            print(f"no improvement ({bad_epochs}/{patience})")
            if bad_epochs >= patience:
                print("Early stopping: no WER improvement")
                break

    print("Training complete.")
    print("Best Val WER:", best_wer)


if __name__ == "__main__":
    main()