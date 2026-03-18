import re

def norm_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()

    d = [[0]*(len(hyp_words)+1) for _ in range(len(ref_words)+1)]
    for i in range(len(ref_words)+1): d[i][0] = i
    for j in range(len(hyp_words)+1): d[0][j] = j

    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )
    return d[-1][-1] / max(1, len(ref_words))