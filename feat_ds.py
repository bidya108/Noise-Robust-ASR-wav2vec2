import os
import torch
from torch.utils.data import Dataset

class FeatASRDataset(Dataset):
    def __init__(self, meta_path: str, feat_clean_dir: str, feat_noisy_dir: str):
        self.items = []
        self.feat_clean_dir = feat_clean_dir
        self.feat_noisy_dir = feat_noisy_dir

        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, text = line.split("|", 1)

                p_clean = os.path.join(feat_clean_dir, utt_id + ".pt")
                p_noisy = os.path.join(feat_noisy_dir, utt_id + ".pt")

                if os.path.exists(p_clean):
                    self.items.append((p_clean, text))
                elif os.path.exists(p_noisy):
                    self.items.append((p_noisy, text))
                else:
                    continue

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, text = self.items[idx]
        feats = torch.load(p)
        return feats, text