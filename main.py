# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pickle
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models import get_models
from optimize import run_optimization
from evaluate import evaluate_model
from plots import generate_all_plots, generate_individual_plots
from utils import create_experiment_dirs, setup_logging, save_checkpoint, load_checkpoint


use_DL_models: bool = False

def run_optuna_pipeline(
    data,
    target_column="target",
    experiment_name=None,
    n_trials=200,
    cv_folds=10,
    test_size=0.2,
    seed=23,
    inference_runs=100,
    use_dl_models: bool = use_DL_models,
):
    # ==== CONFIGURATION ====
    experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path("MODULARIZED_OPTUNA") / experiment_name
    models_to_evaluate = ["mlp", "lstm", "gru"] if use_dl_models else ["lightgbm", "xgboost", "random_forest", "svr", "neural_net"]
    checkpoint_path = save_dir / "checkpoint.pkl"

    # ==== SETUP ====
    save_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"log_modularized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = save_dir / log_filename
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    create_experiment_dirs(save_dir, models_to_evaluate)
    logger = logging.getLogger("optuna_pipeline")
    logger.info("📋 Starting Modularized Optuna Pipeline")

    # ==== CLEAN DATA ====}
    logger.info(f"📦 Original shape: {data.shape}")

    logger.info("🧹 Cleaning dataset (dropping NaNs)")
    data = data.dropna()
    
    



    # ==== SPLIT DATA ====
    split_data_path = save_dir / "split_data.pkl"
    if split_data_path.exists():
        logger.info("🔁 Loading cached train-test split.")
        with open(split_data_path, "rb") as f:
            X_trainval, X_test, y_trainval, y_test = pickle.load(f)
    else:
        logger.info("📐 Performing train-test split.")
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        with open(split_data_path, "wb") as f:
            pickle.dump((X_trainval, X_test, y_trainval, y_test), f)
    logger.info(f"✅ After dropna: {data.shape}")

    # ==== LOAD CHECKPOINT (if any) ====
    checkpoint = load_checkpoint(checkpoint_path)
    completed_models = checkpoint.get("completed_models", [])
    results = checkpoint.get("results", [])

    # ==== RUN OPTIMIZATION AND EVALUATION ====
    full_bar = tqdm(
    total=len(models_to_evaluate),
    desc="📊 Full Pipeline Progress",
    position=0,
    leave=True,
    dynamic_ncols=True,
    )

    eval_runs = inference_runs


    for model_name in models_to_evaluate:
        if model_name in completed_models:
            logger.info(f"✅ Skipping {model_name}, already completed.")
            tqdm.write("🚀 Starting optimization for: {model_name}")
            full_bar.update(1)
            continue

        logger.info(f"🚀 Starting optimization for: {model_name}")
        tqdm.write("🚀 Starting optimization for: {model_name}")

        try:
            with tqdm(total=3, desc=f"⚙️  {model_name} steps", position=1, leave=False) as step_bar:
                step_bar.set_description(f"⚙️  {model_name}: tuning")

                best_model, study, cv_r2, cv_rmse = run_optimization(
                    model_name,
                    save_dir,
                    X_trainval,
                    y_trainval,
                    n_trials,
                    cv_folds,
                )


                step_bar.update(1)

                step_bar.set_description(f"⚙️  {model_name}: evaluating")
                eval_results = evaluate_model(
                    best_model,
                    X_test,
                    y_test,
                    n_inference_runs=eval_runs,
                    save_dir=save_dir / model_name,
                    model_name=model_name,
                )
                generate_individual_plots(
                    best_model,
                    X_test,
                    y_test,
                    save_dir / model_name,
                    model_name,
                )
                step_bar.update(1)

                result_entry = {
                    "Model": model_name,
                    **eval_results,
                    "CV_R2": cv_r2,
                    "CV_RMSE": cv_rmse,
                    "Study": study
                }
                results.append(result_entry)
                completed_models.append(model_name)

                step_bar.set_description(f"⚙️  {model_name}: saving")
                save_checkpoint({"completed_models": completed_models, "results": results}, checkpoint_path)
                with open(save_dir / model_name / "best_model.pkl", "wb") as f:
                    pickle.dump(best_model, f)
                step_bar.update(1)

            logger.info(f"✅ Finished evaluation for: {model_name}")
            tqdm.write("🚀 Starting optimization for: {model_name}")


        except Exception as e:
            logger.exception(f"❌ Error optimizing {model_name}: {e}")
            continue

        full_bar.update(1)


    logger.info("📈 Generating final analysis and plots")
    generate_all_plots(results, save_dir, y_test)
    comparison_dir = Path(save_dir) / "comparison"
    pd.DataFrame(results).to_csv(comparison_dir / "overall_results.csv", index=False)
    with open(comparison_dir / "overall_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # ==== SUMMARY ====
    logger.info("📋 EXECUTIVE SUMMARY:")
    best_model = max(results, key=lambda x: x['R2'])
    fastest_model = min(results, key=lambda x: x['Inference_Time_Mean_ms'])
    most_accurate = min(results, key=lambda x: x['RMSE'])

    logger.info("-" * 40)
    logger.info(f"🏆 Champion Model: {best_model['Model']}")
    logger.info(f"   • R² Score: {best_model['R2']*100:.2f}%")
    logger.info(f"   • RMSE: {best_model['RMSE']:.6f}")
    logger.info(f"   • MAE: {best_model['MAE']:.6f}")
    logger.info(f"   • Inference Time: {best_model['Inference_Time_Mean_ms']:.4f} ± {best_model['Inference_Time_Std_ms']:.4f} ms")

    logger.info("\n📊 KEY INSIGHTS:")
    logger.info("-" * 40)
    logger.info(f"• Total models evaluated: {len(models_to_evaluate)}")
    logger.info(f"• Hyperparameter trials per model: {n_trials}")
    logger.info(f"• Cross-validation folds: {cv_folds}")
    logger.info(f"• Test set size: {len(y_test)} samples ({len(y_test)/len(data)*100:.1f}% of total data)")

    logger.info(f"• Fastest model: {fastest_model['Model']} ({fastest_model['Inference_Time_Mean_ms']:.4f} ms)")
    logger.info(f"• Most accurate model: {most_accurate['Model']} (RMSE: {most_accurate['RMSE']:.6f})")

    logger.info("\n💡 RECOMMENDATIONS:")
    logger.info("-" * 40)
    logger.info(f"✅ The {best_model['Model']} model shows the best overall performance")
    logger.info("   → Recommended for production deployment")
