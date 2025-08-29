# evaluate.py
import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import trange

try:  # pragma: no cover - torch is optional
    import torch
except Exception:  # pragma: no cover
    torch = None


def evaluate_model(
    model,
    X_test,
    y_test,
    *,
    n_inference_runs: int = 100,
    save_dir=None,
    model_name: str = "model",
    use_dl_models: bool = True,
):
    """Evaluate a trained model.

    Parameters
    ----------
    model : estimator
        Trained model implementing ``predict``.
    X_test, y_test : array-like
        Test data and labels.
    n_inference_runs : int, optional
        Number of times to repeat prediction for timing statistics.
    save_dir : Path or str, optional
        If provided, metrics are saved to ``save_dir`` with the pattern
        ``individual_result_<model_name>.csv`` and ``.pkl``.
    model_name : str, optional
        Name used when saving result files.
        use_dl_models : bool, optional
        If True, run predictions within a ``torch.no_grad`` context.
    """

    if use_dl_models and torch is None:
        raise ImportError("PyTorch is required when use_dl_models=True")

    if use_dl_models:
        with torch.no_grad():
            y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Measure inference time
    timings = []
    if use_dl_models:
        with torch.no_grad():
            for _ in trange(n_inference_runs, desc="🕒 Measuring Inference Time"):
                start = time.time()
                _ = model.predict(X_test)
                timings.append((time.time() - start) * 1000)
    else:
        for _ in trange(n_inference_runs, desc="🕒 Measuring Inference Time"):
            start = time.time()
            _ = model.predict(X_test)
            timings.append((time.time() - start) * 1000)

    timing_mean = np.mean(timings)
    timing_std = np.std(timings)

    results = {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Inference_Time_Mean_ms": timing_mean,
        "Inference_Time_Std_ms": timing_std,
    }

    if save_dir is not None:
        from pathlib import Path
        import pandas as pd
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        csv_file = save_path / f"individual_result_{model_name}.csv"
        pkl_file = save_path / f"individual_result_{model_name}.pkl"
        pd.DataFrame([results]).to_csv(csv_file, index=False)
        import pickle
        with open(pkl_file, "wb") as f:
            pickle.dump(results, f)

    return results
