# models.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset



# Optional: PyTorch models
try:  # pragma: no cover - torch is optional
    from pytorch_models import TorchMLPRegressor
except Exception:  # torch is not available
    TorchMLPRegressor = None




class MLPRegressor(BaseEstimator, RegressorMixin):
    """Simple feed-forward neural network regressor using PyTorch."""

    def __init__(
        self,
        hidden_sizes=(64, 64),
        lr=1e-3,
        epochs=100,
        batch_size=32,
        loss="mse",
        verbose=False,
    ):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.verbose = verbose

    def _build_model(self, input_dim):
        layers = []
        prev_dim = input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.model_ = nn.Sequential(*layers)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self._build_model(X_tensor.shape[1])
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.to(self.device_)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.loss == "mse":
            criterion = nn.MSELoss()
        elif self.loss == "mae":
            criterion = nn.L1Loss()
        else:
            raise ValueError("loss must be 'mse' or 'mae'")

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device_), batch_y.to(self.device_)
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss.item():.4f}")
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device_)
            preds = self.model_(X_tensor).cpu().numpy().flatten()
        return preds


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """LSTM-based regressor for sequence data using PyTorch."""

    def __init__(
        self,
        hidden_size=64,
        num_layers=1,
        lr=1e-3,
        epochs=100,
        batch_size=32,
        loss="mse",
        verbose=False,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.verbose = verbose

    def _init_model(self, input_size):
        self.lstm_ = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc_ = nn.Linear(self.hidden_size, 1)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        self._init_model(X_tensor.shape[2])
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_.to(self.device_)
        self.fc_.to(self.device_)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.loss == "mse":
            criterion = nn.MSELoss()
        elif self.loss == "mae":
            criterion = nn.L1Loss()
        else:
            raise ValueError("loss must be 'mse' or 'mae'")

        optimizer = torch.optim.Adam(list(self.lstm_.parameters()) + list(self.fc_.parameters()), lr=self.lr)

        self.lstm_.train()
        self.fc_.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device_), batch_y.to(self.device_)
                optimizer.zero_grad()
                out, _ = self.lstm_(batch_X)
                out = self.fc_(out[:, -1, :])
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss.item():.4f}")
        return self

    def predict(self, X):
        self.lstm_.eval()
        self.fc_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device_)
            out, _ = self.lstm_(X_tensor)
            out = self.fc_(out[:, -1, :])
        return out.cpu().numpy().flatten()


class GRURegressor(BaseEstimator, RegressorMixin):
    """GRU-based regressor for sequence data using PyTorch."""

    def __init__(
        self,
        hidden_size=64,
        num_layers=1,
        lr=1e-3,
        epochs=100,
        batch_size=32,
        loss="mse",
        verbose=False,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.verbose = verbose

    def _init_model(self, input_size):
        self.gru_ = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc_ = nn.Linear(self.hidden_size, 1)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        self._init_model(X_tensor.shape[2])
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru_.to(self.device_)
        self.fc_.to(self.device_)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.loss == "mse":
            criterion = nn.MSELoss()
        elif self.loss == "mae":
            criterion = nn.L1Loss()
        else:
            raise ValueError("loss must be 'mse' or 'mae'")

        optimizer = torch.optim.Adam(list(self.gru_.parameters()) + list(self.fc_.parameters()), lr=self.lr)

        self.gru_.train()
        self.fc_.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device_), batch_y.to(self.device_)
                optimizer.zero_grad()
                out, _ = self.gru_(batch_X)
                out = self.fc_(out[:, -1, :])
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss.item():.4f}")
        return self

    def predict(self, X):
        self.gru_.eval()
        self.fc_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device_)
            out, _ = self.gru_(X_tensor)
            out = self.fc_(out[:, -1, :])
        return out.cpu().numpy().flatten()

def get_models(*, use_DL_models: bool = False, input_dim: int | None = None):
    """Return a dictionary of models to evaluate."""
    models = {
        "lightgbm": lgb.LGBMRegressor(verbose=-1),
        "xgboost": xgb.XGBRegressor(verbosity=0),
        "random_forest": RandomForestRegressor(),
        "svr": SVR(),
        "neural_net": SklearnMLPRegressor(max_iter=1000),
    }

    if use_DL_models:
        if TorchMLPRegressor is None:
            raise ImportError("PyTorch is required for deep learning models")
        if input_dim is None:
            raise ValueError("input_dim must be provided when use_DL_models=True")
        models.update(
            {
                "torch_mlp": TorchMLPRegressor(input_dim=input_dim),
                "mlp": MLPRegressor(),
                "lstm": LSTMRegressor(),
                "gru": GRURegressor(),
            }
        )

    return models