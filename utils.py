# utils.py
import os
import logging
import pickle
from pathlib import Path

import numpy as np


def setup_logging(save_dir):
    log_file = Path(save_dir) / "modularized_optuna.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def create_experiment_dirs(base_path, model_names):
    for name in model_names:
        Path(base_path / name).mkdir(parents=True, exist_ok=True)


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def build_sequences(X, y, seq_len):
    """Create sliding-window sequences.

    Parameters
    ----------
    X : array-like
        Feature data of shape (n_samples, n_features).
    y : array-like
        Target data of length ``n_samples``.
    seq_len : int
        Length of each window.

    Returns
    -------
    tuple of np.ndarray
        ``(X_seq, y_seq)`` where ``X_seq`` has shape ``(n_sequences, seq_len, n_features)``
        and ``y_seq`` has shape ``(n_sequences,)``.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])

    return np.array(X_seq), np.array(y_seq)
