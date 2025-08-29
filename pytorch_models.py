"""PyTorch based regression models.

This module provides lightweight wrappers around PyTorch networks that mimic
the scikit-learn estimator interface. The ``predict`` methods explicitly return
NumPy arrays to integrate smoothly with the existing evaluation utilities.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - torch is optional in the environment
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - fall back if torch is missing
    torch = None
    nn = None
    TensorDataset = DataLoader = object  # type: ignore

from sklearn.base import BaseEstimator, RegressorMixin


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Simple feed-forward neural network regressor implemented in PyTorch.

    Parameters
    ----------
    input_dim : int
        Number of features in the input data.
    hidden_dim : int, default=64
        Size of the hidden layer.
    lr : float, default=1e-3
        Learning rate used by the Adam optimizer.
    epochs : int, default=10
        Number of training epochs.
    batch_size : int, default=32
        Mini-batch size used during training.
    device : str or torch.device, optional
        Device on which to run the model. If ``None`` (default) it will
        automatically select ``"cuda"`` when available otherwise ``"cpu"``.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
        device: str | None = None,
    ):
        if torch is None:  # pragma: no cover - handled in runtime checks
            raise ImportError("PyTorch is required for TorchMLPRegressor")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # --- scikit-learn compatible methods ---------------------------------
    def fit(self, X, y):  # noqa: D401 - standard scikit-learn signature
        """Fit the neural network to the training data."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        return self

    def predict(self, X):  # noqa: D401 - standard scikit-learn signature
        """Predict targets for ``X`` and return NumPy arrays."""
        if self.model is None:  # pragma: no cover - safety check
            raise RuntimeError("Model not fitted")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
            preds = self.model(X_tensor).cpu().numpy().ravel()
        return preds

