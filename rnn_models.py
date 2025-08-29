# rnn_models.py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import build_sequences


class _BaseRNNRegressor(nn.Module):
    def __init__(self, rnn_cls, input_size, hidden_size=64, num_layers=1, seq_len=10,
                 lr=1e-3, batch_size=32, epochs=10):
        super().__init__()
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.seq_len = seq_len
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)

    def _make_loader(self, X, y):
        X_seq, y_seq = build_sequences(X, y, self.seq_len)
        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def fit(self, X, y, X_val=None, y_val=None):
        train_loader = self._make_loader(X, y)
        val_loader = (
            self._make_loader(X_val, y_val) if X_val is not None and y_val is not None else None
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            self.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self(xb), yb)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        _ = criterion(self(xb), yb)
        return self

    def predict(self, X):
        self.eval()
        X_seq, _ = build_sequences(X, np.zeros(len(X)), self.seq_len)
        data = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self(data).cpu().numpy()
        return preds


class LSTMRegressor(_BaseRNNRegressor):
    def __init__(self, input_size, **kwargs):
        super().__init__(nn.LSTM, input_size, **kwargs)


class GRURegressor(_BaseRNNRegressor):
    def __init__(self, input_size, **kwargs):
        super().__init__(nn.GRU, input_size, **kwargs)
