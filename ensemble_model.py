import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

class EnsembleModel:
    def __init__(self):
        self.model_xgb_time = xgb.XGBRegressor(objective='reg:squarederror')
        self.model_lgb_time = lgb.LGBMRegressor()
        self.model_xgb_compute = xgb.XGBRegressor(objective='reg:squarederror')
        self.model_lgb_compute = lgb.LGBMRegressor()
        self.meta_model_time = Ridge(alpha=1.0)
        self.meta_model_compute = Ridge(alpha=1.0)

    def fit(self, X, y_time, y_compute):
        # Split into train/meta
        X_train, X_meta, y_time_train, y_time_meta = train_test_split(X, y_time, test_size=0.2, random_state=42)
        _, _, y_compute_train, y_compute_meta = train_test_split(X, y_compute, test_size=0.2, random_state=42)

        # Base models for time
        self.model_xgb_time.fit(X_train, y_time_train)
        self.model_lgb_time.fit(X_train, y_time_train)

        pred_xgb_time = self.model_xgb_time.predict(X_meta)
        pred_lgb_time = self.model_lgb_time.predict(X_meta)

        meta_X_time = np.vstack([pred_xgb_time, pred_lgb_time]).T
        self.meta_model_time.fit(meta_X_time, y_time_meta)

        # Base models for compute
        self.model_xgb_compute.fit(X_train, y_compute_train)
        self.model_lgb_compute.fit(X_train, y_compute_train)

        pred_xgb_compute = self.model_xgb_compute.predict(X_meta)
        pred_lgb_compute = self.model_lgb_compute.predict(X_meta)

        meta_X_compute = np.vstack([pred_xgb_compute, pred_lgb_compute]).T
        self.meta_model_compute.fit(meta_X_compute, y_compute_meta)

    def predict(self, X):
        pred_time_xgb = self.model_xgb_time.predict(X)
        pred_time_lgb = self.model_lgb_time.predict(X)
        meta_time_input = np.vstack([pred_time_xgb, pred_time_lgb]).T
        final_time = self.meta_model_time.predict(meta_time_input)

        pred_compute_xgb = self.model_xgb_compute.predict(X)
        pred_compute_lgb = self.model_lgb_compute.predict(X)
        meta_compute_input = np.vstack([pred_compute_xgb, pred_compute_lgb]).T
        final_compute = self.meta_model_compute.predict(meta_compute_input)

        return final_time, final_compute
