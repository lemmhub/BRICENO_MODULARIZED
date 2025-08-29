import optuna
from sklearn.model_selection import cross_val_score
from models import get_models
from tqdm import tqdm
import pickle
from pathlib import Path


def suggest_mlp(trial):
    """Suggest hyperparameters for a basic MLP model."""
    return {
        "n_layers": trial.suggest_int("mlp_n_layers", 1, 3),
        "hidden_size": trial.suggest_int("mlp_hidden_size", 16, 256),
        "dropout": trial.suggest_float("mlp_dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("mlp_learning_rate", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("mlp_batch_size", [32, 64, 128]),
        "epochs": trial.suggest_int("mlp_epochs", 10, 200),
        "loss": trial.suggest_categorical("mlp_loss", ["mse", "mae"]),
    }

def suggest_torch_mlp(trial):
    """Suggest hyperparameters for TorchMLPRegressor."""
    return {
        "hidden_dim": trial.suggest_int("torch_mlp_hidden_dim", 16, 256),
        "lr": trial.suggest_float("torch_mlp_lr", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("torch_mlp_batch_size", [32, 64, 128]),
        "epochs": trial.suggest_int("torch_mlp_epochs", 10, 200),
    }



def suggest_lstm(trial):
    """Suggest hyperparameters for an LSTM model."""
    return {
        "n_layers": trial.suggest_int("lstm_n_layers", 1, 3),
        "hidden_size": trial.suggest_int("lstm_hidden_size", 16, 256),
        "dropout": trial.suggest_float("lstm_dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("lstm_learning_rate", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("lstm_batch_size", [32, 64, 128]),
        "epochs": trial.suggest_int("lstm_epochs", 10, 200),
        "loss": trial.suggest_categorical("lstm_loss", ["mse", "mae"]),
    }


def suggest_gru(trial):
    """Suggest hyperparameters for a GRU model."""
    return {
        "num_layers": trial.suggest_int("gru_n_layers", 1, 3),
        "hidden_size": trial.suggest_int("gru_hidden_size", 16, 256),
        "dropout": trial.suggest_float("gru_dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("gru_learning_rate", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("gru_batch_size", [32, 64, 128]),
        "epochs": trial.suggest_int("gru_epochs", 10, 200),
        "loss": trial.suggest_categorical("gru_loss", ["mse", "mae"]),
    }


def run_optimization(
    model_name,
    save_dir,
    X,
    y,
    n_trials,
    cv,
    logger,
    use_dl_models: bool = True,
):
    """Run hyperparameter optimization for a given model.

    Parameters
    ----------
    model_name : str
        Identifier of the model to optimize.
    save_dir : Path or str
        Directory to save optimization artifacts.
    X, y : array-like
        Training data and targets.
    n_trials : int
        Number of Optuna trials.
    cv : int
        Number of cross-validation folds.
    logger : logging.Logger
        Logger for recording progress messages.
    use_dl_models : bool, optional
        Include deep-learning models when True.
    """

    pbar = tqdm(total=n_trials, desc=f"🧪 {model_name} tuning", dynamic_ncols=True, leave=True)

    def update_progress_bar(study, trial):
        best_value = study.best_value
        best_params = study.best_trial.params
        trial_number = trial.number
        best_r2 = max((t.user_attrs.get("r2_mean", float("-inf")) for t in study.trials), default=float("nan"))
        study.set_user_attr("best_r2", best_r2)
        pbar.update(1)
        message = (
            f"🔁 Trial {trial_number}: Best RMSE={best_value:.6f}, Best R2={best_r2:.4f}, Best Params={best_params}"
        )
        tqdm.write(message)
        logger.info(message)

    def objective(trial):
        model_dict = get_models(
            use_dl_models=use_dl_models,
            input_dim=X.shape[1] if use_dl_models else None,
        )

        dl_params_map = {
            "torch_mlp": suggest_torch_mlp,
            "mlp": suggest_mlp,
            "lstm": suggest_lstm,
            "gru": suggest_gru,
        }

        if use_dl_models and model_name in dl_params_map:
            params = dl_params_map[model_name](trial)
        elif model_name == "lightgbm":

            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "verbose": -1,
            }
        elif model_name == "xgboost":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "verbosity": 0,
            }
        elif model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "verbose": 0,
            }
        elif model_name == "svr":
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
            params = {
                "C": trial.suggest_float("C", 0.1, 100),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
                "kernel": kernel,
            }
            if kernel == "rbf":
                params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        elif model_name == "neural_net":
            solver = trial.suggest_categorical("solver", ["adam", "lbfgs"])
            params = {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes", [(64,), (128,), (64, 64), (128, 64)]
                ),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "solver": solver,
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1),
                "max_iter": trial.suggest_int("max_iter", 200, 1000),
                "verbose": False,
            }
            if solver == "adam":
                params["batch_size"] = trial.suggest_categorical("batch_size", ["auto", 32, 64, 128])
        else:
            raise ValueError("Unsupported model")

        model = model_dict[model_name].set_params(**params)

        r2_scores = cross_val_score(model, X, y, scoring="r2", cv=cv)
        r2_mean = r2_scores.mean()
        

        rmse_scores = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)
        rmse_mean = -rmse_scores.mean()

        trial.set_user_attr("r2_mean", r2_mean)
        trial.set_user_attr("rmse_mean", rmse_mean)
        if r2_mean < 0.90:
            return float("inf")

        return rmse_mean

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[update_progress_bar])
    pbar.close()

    best_model = get_models(
        use_dl_models=use_dl_models, input_dim=X.shape[1] if use_dl_models else None
    )[model_name].set_params(**study.best_params)

    best_model.fit(X, y)
    study.user_attrs["final_model"] = best_model

    cv_r2 = cross_val_score(best_model, X, y, scoring="r2", cv=cv).mean()
    cv_rmse = -cross_val_score(best_model, X, y, scoring="neg_root_mean_squared_error", cv=cv).mean()

    model_dir = Path(save_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    study_path = model_dir / f"{model_name}_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)

    return best_model, study, cv_r2, cv_rmse
