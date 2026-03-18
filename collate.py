import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, tokenizer):
    feats_list, texts = zip(*batch)

    input_lengths = torch.tensor([x.size(0) for x in feats_list], dtype=torch.long)
    feats_padded = pad_sequence(feats_list, batch_first=True)

    targets = []
    target_lengths = []
    for t in texts:
        ids = tokenizer.encode(t)
        targets.extend(ids)
        target_lengths.append(len(ids))

    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return feats_padded, targets, input_lengths, target_lengths, list(texts)