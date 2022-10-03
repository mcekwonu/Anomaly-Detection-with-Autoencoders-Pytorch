import torch
import torch.nn as nn


class GRU(nn.Module):
    name = __qualname__

    def __init__(self, seq_len, n_features, embedding_dim=128, num_layers=2, bidirectional=False):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.linear = nn.Linear(in_features=embedding_dim, out_features=n_features)

    def forward(self, x):
        x = x.reshape(1, self.seq_len, self.n_features)

        out, _ = self.gru(x)
        out = out[:, -1, :]

        return self.linear(out)

    def model_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nTrainable parameters: {total_params}")
