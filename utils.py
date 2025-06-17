# utils.py
import os
import logging
import pickle
from pathlib import Path


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
