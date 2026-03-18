import torch
import torch.nn as nn

class SimpleASRModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, vocab_size=29):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x) 
        x = x.transpose(1, 2) 
        x, _ = self.rnn(x) 
        x = self.fc(x) 
        return x