# model.py
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from crepes import ConformalRegressor


def train_model(X, y):
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    return model, X_cal, y_cal

def forecast(model, future_dates, model_columns = ['ordinal_datetime', 'month', 'week', 'sales_lag_1y']):
    y_forecast = model.predict(future_dates[model_columns]).flatten()
    return y_forecast

def calculate_intervals(model, X_cal, y_cal, future_dates, model_columns = ['ordinal_datetime', 'month', 'week', 'sales_lag_1y']):
    y_hat_cal = model.predict(X_cal[model_columns])
    X_cal["y_cal"] = y_hat_cal
    X_cal["residuals_cal"] = y_cal - y_hat_cal

    model_cps_std = ConformalRegressor().fit(residuals=X_cal['residuals_cal'].values)
    intervals = model_cps_std.predict(y_hat=future_dates['y_forecast'].values, confidence=0.80)
    intervals_array = np.array(intervals)
    intervals_df = pd.DataFrame(intervals_array, columns=['lower_bound', 'upper_bound'])
    future_dates['y_forecast-lo-80'] = intervals_df['lower_bound'].reset_index(drop=True)
    future_dates['y_forecast-hi-80'] = intervals_df['upper_bound'].reset_index(drop=True)

    return future_dates
