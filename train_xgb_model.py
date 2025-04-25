# train_xgb_model.py
import pandas as pd
import numpy as np
import joblib
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# === Load Data ===
df = pd.read_excel("/Users/linyunhao/Documents/GitHub/Dashboard-Almanax/AI_Agent_Records_Final_10327.xlsx")

# === Define Features and Target ===
X = df[[
    'wallet_access_status', 'pii_handling_status', 'spending_limits',
    'SLOC', 'LLOC', 'CLOC', 'NF', 'WMC', 'NL', 'NLE', 'NUMPAR', 'NOS',
    'DIT', 'NOA', 'NOD', 'CBO', 'NA', 'NOI',
    'Avg_McCC', 'Avg_NL', 'Avg_NLE', 'Avg_NUMPAR', 'Avg_NOS', 'Avg_NOI'
]]
y = df['risk_score']

# === Preprocessing Pipeline ===
cat_cols = ['wallet_access_status', 'pii_handling_status']
num_cols = [col for col in X.columns if col not in cat_cols]

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

# === Model Pipeline ===
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, random_state=42))
])

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
xgb_model.fit(X_train, y_train)

# === Evaluate ===
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("XGBoost Model Performance:")
print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")

# === Save Model ===
joblib.dump(xgb_model, "/Users/linyunhao/Documents/GitHub/Dashboard-Almanax/xgboost_model.pkl")

# === SHAP Explainer ===
X_transformed = xgb_model.named_steps['preprocessor'].fit_transform(X)
explainer = shap.Explainer(xgb_model.named_steps['regressor'], X_transformed)
joblib.dump(explainer, "/Users/linyunhao/Documents/GitHub/Dashboard-Almanax/shap_explainer.pkl")

print("/Users/linyunhao/Documents/GitHub/Dashboard-Almanax")
