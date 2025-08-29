import argparse
from tqdm import tqdm
from main import run_optuna_pipeline

def run_main_pipeline(experiment_name=None, trials=200, cv=10, test_size=0.2, inference_runs=100, use_dl_models=True):
    import pandas as pd

    files = ["DS_OCV_1.xlsx", "DS_OCV_2.xlsx", "DS_OCV_3.xlsx"]

    # Persistent progress bar for loading and running the pipeline
    with tqdm(total=1, desc="📦 Preparing and Running Pipeline", dynamic_ncols=True, leave=True) as pbar:
        dfs = [pd.read_excel(f).iloc[:, 1:] for f in files]  # drop first column (row number)
        df_combined = pd.concat(dfs, ignore_index=True)
        target_column = df_combined.columns[-1]  # last column is the target by default

        tqdm.write("🔍 Starting Optuna pipeline...")
        run_optuna_pipeline(
            data=df_combined,
            target_column=target_column,
            experiment_name=experiment_name,
            n_trials=trials,
            cv_folds=cv,
            test_size=test_size,
            inference_runs=inference_runs,
            use_dl_models=use_dl_models,
        )
        pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Modularized Optuna Experiment")
    parser.add_argument("--experiment_name", type=str, default=None, help="Optional experiment name override")
    parser.add_argument("--trials", type=int, default=200, help="Number of Optuna trials per model")
    parser.add_argument("--cv", type=int, default=10, help="Number of cross-validation folds")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set split ratio")
    parser.add_argument("--inference_runs", type=int, default=100, help="Number of repetitions for inference timing")
    parser.add_argument("--use_dl_models", action="store_true", help="Use deep learning models (mlp, lstm, gru)")
    args = parser.parse_args()

    run_main_pipeline(
        experiment_name=args.experiment_name,
        trials=args.trials,
        cv=args.cv,
        test_size=args.test_size,
        inference_runs=args.inference_runs,
        use_dl_models=args.use_dl_models,
    )
