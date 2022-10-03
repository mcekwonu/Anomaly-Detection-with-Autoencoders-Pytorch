import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, num_layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim // 2

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, hidden_states = self.lstm(x)

        return x[:, -1, :]


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=2, num_layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = input_dim // 2
        self.n_features = n_features

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=n_features)

    def forward(self, x):
        x = x.reshape((1, -1, self.input_dim))
        x = x.repeat(1, self.seq_len, 1)

        x, hidden_states = self.lstm(x)

        return self.linear(x)


class BiLSTMAE(nn.Module):
    """Bidirectional LSTM AutoEncoder for anomaly detection"""
    name = __qualname__

    def __init__(self, seq_len, n_features, embedding_dim=64, num_layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = Encoder(self.seq_len, self.n_features, self.embedding_dim, num_layers)
        self.decoder = Decoder(self.seq_len, self.embedding_dim, self.n_features, num_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def model_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nTrainable parameters: {total_params}")