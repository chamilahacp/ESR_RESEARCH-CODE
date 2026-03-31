# =========================================================
# FINAL THESIS UNIFIED RUNNER: RF, GBR, XGB, SVR
# =========================================================
import numpy as np
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# 1. Setup & Data Loading
drive.mount('/content/drive')
CSV_PATH = "/content/drive/My Drive/x/p/P5p.csv"
SAVE_DIR = "/content/drive/My Drive/x/p"
TARGET = "A"
PREDICTORS = ["N6", "S4", "T4", "P5", "O5", "R4", "H5", "S"]

df = pd.read_csv(CSV_PATH)
data = df[[TARGET] + PREDICTORS].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
X, y = data[PREDICTORS], data[TARGET]

# One fixed split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 2. Define All Models with Tuned Hyperparameters
models = {
    "Random_Forest": RandomForestRegressor(
        n_estimators=198, max_depth=11, max_features="sqrt", 
        min_samples_leaf=2, min_samples_split=11, bootstrap=False, random_state=42
    ),
    "Gradient_Boosting": GradientBoostingRegressor(
        n_estimators=221, learning_rate=0.0115, max_depth=5, 
        min_samples_split=9, subsample=0.956, max_features="sqrt", random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=335, learning_rate=0.0462, max_depth=4, 
        min_child_weight=9, subsample=0.836, colsample_bytree=0.831, random_state=42
    ),
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(C=44.16, epsilon=1.216, kernel="rbf"))
    ])
}

# 3. The Big Loop: Train, Predict, Evaluate, Plot, Save
for name, model in models.items():
    print(f"\n{'='*30}\nRUNNING MODEL: {name}\n{'='*30}")
    
    # Train & Predict
    model.fit(X_train, y_train)
    y_tr_hat, y_te_hat = model.predict(X_train), model.predict(X_test)
    
    # Metrics Calculation
    for set_name, y_true, y_pred in [("Train", y_train, y_tr_hat), ("Test", y_test, y_te_hat)]:
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f"   {set_name} | R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

    # Plot Observed vs Predicted
    plt.figure(figsize=(5,5))
    plt.scatter(y_train, y_tr_hat, label="Train", alpha=0.6)
    plt.scatter(y_test, y_te_hat, label="Test", color="orangered", alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.title(f"{name}: Obs vs Pred")
    plt.xlabel("Observed APTI"); plt.ylabel("Predicted APTI")
    plt.legend(); plt.show()

    # Save Predictions CSV
    out = data.copy()
    out["Pred"] = np.nan
    out.loc[X_train.index, "Pred"] = y_tr_hat
    out.loc[X_test.index, "Pred"] = y_te_hat
    out.to_csv(f"{SAVE_DIR}/{name}_final_preds.csv", index=False)

    # Explainability (PDP & SHAP) - Only for Tree Models (SVR needs different SHAP)
    if name != "SVR":
        # SHAP Summary
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        print(f"Generating SHAP for {name}...")
        shap.summary_plot(shap_values, X_test, show=True)
