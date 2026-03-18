import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)

class ConvSubsample(nn.Module):
    def __init__(self, in_feats: int = 80, d_model: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_feats, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x) 
        x = x.transpose(1, 2) 
        return x


class TransformerCTC(nn.Module):
    def __init__(
        self,
        in_feats: int = 80,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        vocab_size: int = 29,
    ):
        super().__init__()

        self.subsample = ConvSubsample(in_feats=in_feats, d_model=d_model)
        self.posenc = PositionalEncoding(d_model=d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True, 
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.subsample(x) 
        x = self.posenc(x)
        x = self.encoder(x) 
        x = self.fc(x) 
        return x