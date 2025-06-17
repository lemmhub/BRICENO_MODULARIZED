# models.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb


def get_models():
    return {
        "lightgbm": lgb.LGBMRegressor(),
        "xgboost": xgb.XGBRegressor(),
        "random_forest": RandomForestRegressor(),
        "svr": SVR(),
        "neural_net": MLPRegressor(max_iter=1000)
    }
