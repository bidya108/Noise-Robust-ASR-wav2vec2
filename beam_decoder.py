import numpy as np

try:
    from pyctcdecode import build_ctcdecoder
except ImportError:
    build_ctcdecoder = None


def build_decoder(tokenizer):
    if build_ctcdecoder is None:
        print("pyctcdecode not installed. Beam decoding disabled.")
        return None

    alphabet = []
    for i in range(len(tokenizer.idx2char)):
        tok = tokenizer.idx2char[i]

        if i == 0 or tok == "<blank>":
            continue

        if not isinstance(tok, str) or len(tok) != 1:
            print(f"Skipping non-1-char token in vocab: {tok!r}")
            continue

        alphabet.append(tok)

    print("Tokenizer vocab size (incl blank):", len(tokenizer.idx2char)) 
    print("Decoder alphabet size (no blank):", len(alphabet)) 

    return build_ctcdecoder(alphabet)


def beam_decode_batch(decoder, out_log, out_lens, beam_width=20):
    if decoder is None:
        raise RuntimeError("Decoder is None. Install pyctcdecode or disable USE_BEAM.")

    out_log = out_log.detach().cpu().float().numpy()
    out_lens = out_lens.detach().cpu().numpy().astype(int)

    texts = []

    for lp, L in zip(out_log, out_lens):
        lp = lp[:L, :] 
        lp = lp[:, 1:] 

        text = decoder.decode(lp, beam_width=beam_width)
        texts.append(text)

    return texts