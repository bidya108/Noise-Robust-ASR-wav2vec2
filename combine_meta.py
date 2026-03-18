CLEAN = "datasets/metadata_10k_clean.txt"
NOISY = "datasets/metadata_10k_noisy.txt"
OUT = "datasets/meta_20k.txt"

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

clean = read_lines(CLEAN)
noisy = read_lines(NOISY)

combined = clean + noisy

with open(OUT, "w", encoding="utf-8") as f:
    for ln in combined:
        f.write(ln + "\n")

print("Clean:", len(clean))
print("Noisy:", len(noisy))
print("Total:", len(combined))
print("Saved:", OUT)