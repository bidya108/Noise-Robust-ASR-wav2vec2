import os
import random
from src.utils.text import normalize_transcript

def collect_librispeech_metadata(librispeech_root: str):
    items = []
    for root, _, files in os.walk(librispeech_root):
        for f in files:
            if f.endswith(".trans.txt"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as fp:
                    for line in fp:
                        parts = line.strip().split(" ", 1)
                        if len(parts) != 2:
                            continue
                        utt_id, text = parts
                        text = normalize_transcript(text)
                        if len(text) == 0:
                            continue
                        items.append((utt_id, text))
    return items

def write_metadata(items, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for utt_id, text in items:
            f.write(f"{utt_id}|{text}\n")

if __name__ == "__main__":
    random.seed(42)

    librispeech_root = "datasets/librispeech/train-clean-100"
    items = collect_librispeech_metadata(librispeech_root)

    random.shuffle(items)
    n = len(items)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_items = items[:n_train]
    val_items   = items[n_train:n_train+n_val]
    test_items  = items[n_train+n_val:]

    write_metadata(items,      "datasets/metadata.txt")
    write_metadata(train_items,"datasets/metadata_train.txt")
    write_metadata(val_items,  "datasets/metadata_val.txt")

    print("Total:", len(items))
    print("Train:", len(train_items))
    print("Val:", len(val_items))
    print("Test:", len(test_items))