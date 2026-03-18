import random

IN_PATH = "datasets/metadata_train.txt"
OUT_PATH = "datasets/metadata_10k_clean.txt"
N = 10000
SEED = 42

random.seed(SEED)

with open(IN_PATH, "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

random.shuffle(lines)
lines = lines[:N]

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for ln in lines:
        f.write(ln + "\n")

print("Wrote:", OUT_PATH, "count:", len(lines))
