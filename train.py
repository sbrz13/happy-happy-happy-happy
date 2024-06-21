from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
# Read Excel file specifying the engine

# 处理缺失值（例如，用均值填充）

# Assuming Xtrain, ytrain, Xtest, ytest are predefined datasets for training and testing
train = pd.read_csv('happiness_train_abbr.csv', parse_dates=['survey_time'], encoding='utf-8')
test = pd.read_csv("happiness_test_abbr.csv", parse_dates=["survey_time"], encoding='latin-1')
X = train.drop(columns=['happiness'])
X = train.drop(columns=['survey_time'])
y = train['happiness']
train.fillna(0, inplace=True)
train = train.loc[train['happiness'] != -8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a default XGBoost model
xgb_default = XGBRegressor()
xgb_default.fit(X_train, y_train)
predictions_xgb = xgb_default.predict(X_test)

# Calculate and print Mean Squared Error (MSE) and R^2 score for XGBoost
mse_xgb = mean_squared_error(y_test, predictions_xgb)
r2_xgb = r2_score(y_test, predictions_xgb)
print(f'XGBOOST Default MSE: {mse_xgb:.4f}')
print(f'XGBOOST Default R^2: {r2_xgb:.4f}')

# Train a default LightGBM model
lgbm_default = LGBMRegressor()
lgbm_default.fit(X_train, y_train)
predictions_lgbm = lgbm_default.predict(X_test)

# Calculate and print MSE and R^2 score for LightGBM
mse_lgbm = mean_squared_error(y_test, predictions_lgbm)
r2_lgbm = r2_score(y_test, predictions_lgbm)
print(f'LGBM Default MSE: {mse_lgbm:.4f}')
print(f'LGBM Default R^2: {r2_lgbm:.4f}')

# Fine-tune XGBoost with specific hyperparameters
xgb_tuned = XGBRegressor(
    max_depth=6,
    learning_rate=0.01,
    n_estimators=3000,
    silent=False,
    objective='reg:squarederror',
    booster='gbtree',
    n_jobs=-1,
    gamma=5.4,
    min_child_weight=6,
    subsample=0.8,
    colsample_bytree=1,
    reg_lambda=1.39,
    random_state=7  # 'seed' is deprecated and replaced by 'random_state'
)

xgb_tuned.fit(X_train, y_train)
predictions_tuned = xgb_tuned.predict(X_test)

# Calculate and print MSE and R^2 score for the tuned XGBoost model
mse_tuned = mean_squared_error(y_test, predictions_tuned)
r2_tuned = r2_score(y_test, predictions_tuned)
print(f'Tuned XGBoost MSE: {mse_tuned:.4f}')
print(f'Tuned XGBoost R^2: {r2_tuned:.4f}')

# Save the best model
xgb_tuned.save_model('The_Best_Model.json')  # It's good practice to specify the file extension clearly

# Assuming X_test is the same as Xtest
predictions_best = xgb_tuned.predict(X_test)