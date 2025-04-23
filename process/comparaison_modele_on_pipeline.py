
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from preprocess_data import PreprocessData
def mean_squared_error(y_true, y_pred, squared=False):
    return np.sqrt((y_true - y_pred) ** 2).mean()
# Chargement des données
X_train_full = pd.read_csv('data/X_train_6ZIKlTY.csv')
y_train_full = pd.read_csv('data/y_train_lXj6X5y.csv')['target']
X_test = pd.read_csv('data/X_test_oiZ2ukx.csv')

# Application du prétraitement
preprocessor = PreprocessData(X_train_full, y_train_full, X_test)
preprocessor.process_transformation()
X_train, y_train, X_test = preprocessor.get_data()

# Préparation des données pour imputation de 'on'
features_on = ['age', 'sexM', 'cohort', 'off', 'age_at_diagnosis', 
               'est_LRRK2+', 'est_GBA+', 'est_OTHER+']
df_on = X_train.dropna(subset=features_on + ['on'])
X = df_on[features_on]
y = df_on['on']

# Modèles à évaluer
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
}

results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    mae_scores, rmse_scores, r2_scores = [], [], []

    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)

        mae_scores.append(mean_absolute_error(y_val_fold, y_pred))
        rmse_scores.append(mean_squared_error(y_val_fold, y_pred, squared=False))
        r2_scores.append(r2_score(y_val_fold, y_pred))

    results.append({
        "Modèle": name,
        "MAE": round(np.mean(mae_scores), 2),
        "RMSE": round(np.mean(rmse_scores), 2),
        "R²": round(np.mean(r2_scores), 2),
        "Learning rate": "—" if name != "XGBoost" else 0.1,
        "n_estimators": "—" if name in ["LinearRegression", "Ridge"] else 100
    })

# Affichage des résultats
results_df = pd.DataFrame(results)
print("\nComparaison des modèles pour l'imputation de 'on':")
print(results_df)
