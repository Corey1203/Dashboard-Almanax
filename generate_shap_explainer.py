# generate_shap_explainer.py

from xgboost import XGBClassifier
import shap
import joblib
import pandas as pd

# Load your training data (adjust this path as necessary)
# For demonstration purposes, assume a small mock dataset
X = pd.read_csv("AI_Agent_Records_Final_10327.csv").dropna().select_dtypes(include='number')

# Load model
model = joblib.load("xgboost_model.pkl")

# Create SHAP explainer in Docker environment
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, "shap_explainer.pkl")

print("SHAP explainer saved as shap_explainer.pkl inside the container.")