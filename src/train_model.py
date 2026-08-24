import os
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.load_data import load_data
from src.preprocess import preprocess

# 1. Load Data & Preprocess
df = load_data()
X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

# 2. Compare Baseline Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "R2": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
    }

results_df = pd.DataFrame(results).T
print("--- Baseline Model Comparison ---")
print(results_df.round(4))

# 3. Fit Best Model Directly (Fast for Cloud)
best_model = models["Random Forest"]

# 4. Save Model & Scaler Artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\nSaved best_model.pkl and scaler.pkl to models/")