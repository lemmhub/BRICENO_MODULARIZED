# evaluate.py
import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import trange


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    # Measure inference time
    timings = []
    for _ in trange(200, desc="🕒 Measuring Inference Time"):
        start = time.time()
        _ = model.predict(X_test)
        timings.append((time.time() - start) * 1000)

    timing_mean = np.mean(timings)
    timing_std = np.std(timings)

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Inference_Time_Mean_ms": timing_mean,
        "Inference_Time_Std_ms": timing_std,
    }
