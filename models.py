# models.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb


# Optional: PyTorch models
try:  # pragma: no cover - torch is optional
    from pytorch_models import TorchMLPRegressor
except Exception:  # torch is not available
    TorchMLPRegressor = None


def get_models(*, use_DL_models: bool = False, input_dim: int | None = None):
    """Return a dictionary of models to evaluate.

    Parameters
    ----------
    use_DL_models : bool, optional
        Include deep learning models implemented with PyTorch if ``True`` and
        PyTorch is available. Defaults to ``False``.
    input_dim : int, optional
        Number of features in the input data. Required when
        ``use_DL_models`` is ``True`` so that PyTorch models can be
        constructed correctly.
    """

    models = {
        "lightgbm": lgb.LGBMRegressor(),
        "xgboost": xgb.XGBRegressor(),
        "random_forest": RandomForestRegressor(),
        "svr": SVR(),
        "neural_net": MLPRegressor(max_iter=1000),
    }

    if use_DL_models:
        if TorchMLPRegressor is None:
            raise ImportError("PyTorch is required for deep learning models")
        if input_dim is None:
            raise ValueError("input_dim must be provided when use_DL_models=True")
        models["torch_mlp"] = TorchMLPRegressor(input_dim=input_dim)

    return models
