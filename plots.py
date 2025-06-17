# plots.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
from matplotlib.patches import Patch
from pathlib import Path


def generate_individual_plots(model, X_test, y_test, save_dir, model_name):
    """Generate and save plots for a single trained model."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Actual vs Predicted
    y_pred = model.predict(X_test)
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--k")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted: {model_name}")
    plt.savefig(save_dir / f"actual_vs_pred_{model_name}.png")
    plt.close()

    # Residual Distribution
    residuals = y_test - y_pred
    plt.figure()
    sns.histplot(residuals, kde=True)
    plt.title(f"Residual Distribution: {model_name}")
    plt.xlabel("Residual")
    plt.savefig(save_dir / f"residuals_{model_name}.png")
    plt.close()


def generate_all_plots(results, save_dir, y_true):
    """Generate comparison plots across all models."""
    save_dir = Path(save_dir) / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    # Radar chart
    radar_labels = ["R2", "RMSE", "MAE", "Inference_Time_Mean_ms"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    for _, row in df.iterrows():
        values = [row[label] for label in radar_labels]
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, label=row['Model'])
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels)
    ax.set_title("Model Performance Radar Chart")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.savefig(save_dir / "radar_chart.png")
    plt.close()

    # Heatmap
    heatmap_df = df.set_index("Model")[radar_labels]
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu")
    plt.title("Model Comparison Heatmap")
    plt.savefig(save_dir / "heatmap.png")
    plt.close()

    # Actual vs Predicted for each model
    for entry in results:
        model = entry['Study'].user_attrs.get("final_model")
        if model is None:
            continue
        y_pred = model.predict(y_true.index.to_frame())
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--k')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted: {entry['Model']}")
        plt.savefig(save_dir / f"actual_vs_pred_{entry['Model']}.png")
        plt.close()

    # CV Score Distribution Placeholder (since CV results not stored)
    # Prediction Distribution Placeholder (since model predictions not stored)
    # Optional: implement if you log prediction arrays
