class SimpleTokenizer:
    def __init__(self):
        vocab = ["<blank>", " "] + list("abcdefghijklmnopqrstuvwxyz'")

        self.idx2char = {i: c for i, c in enumerate(vocab)}
        self.char2idx = {c: i for i, c in enumerate(vocab)}

    def encode(self, text: str):
        text = text.lower()
        text = "".join([c for c in text if c in self.char2idx])
        return [self.char2idx[c] for c in text]

    def decode_ctc_greedy(self, ids):
        out = []
        prev = None
        for i in ids:
            if i != prev and i != 0:
                out.append(self.idx2char[i])
            prev = i
        text = "".join(out)
        return " ".join(text.split())