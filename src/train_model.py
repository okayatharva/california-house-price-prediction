import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from load_data import load_data
from preprocess import preprocess

# 1. Load & Preprocess
df = load_data()
X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess(df)

# 2. Compare Baseline Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

print("--- Baseline Model Comparison ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name:20s} | R2: {r2:.4f} | RMSE: {rmse:.4f}")

# 3. Fine-Tune Best Model (Random Forest)
print("\n--- Running GridSearchCV for Random Forest ---")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# 4. Save Model & Scaler Artifacts
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\nSaved best_model.pkl and scaler.pkl to models/")