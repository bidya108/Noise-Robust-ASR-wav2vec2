import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) 
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T]

class ConvSubsample(nn.Module):
    def __init__(self, in_feats=80, d_model=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_feats, d_model, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.conv(x)
        x = x.transpose(1, 2) 
        return x

class TransformerASR(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size,
        d_model=256,
        nhead=4,
        num_layers=6,
        dim_ff=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.subsample = ConvSubsample(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, src_key_padding_mask=None):
        x = self.subsample(x) 
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask) 
        x = self.fc(x) 
        return x