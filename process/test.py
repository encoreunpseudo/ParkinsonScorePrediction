import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Import de notre classe de prétraitement
from preprocess_data import PreprocessData

# Chemins des fichiers
X_TRAIN_PATH = 'data/X_train_6ZIKlTY.csv'
Y_TRAIN_PATH = 'data/y_train_lXj6X5y.csv'
X_TEST_PATH = 'data/X_test_oiZ2ukx.csv'

def main():
    """
    Script de test pour valider notre prétraitement et notre modèle
    en simulant le contexte d'évaluation du challenge
    """
    # Chargement des données
    print("Chargement des données...")
    X_train_full = pd.read_csv(X_TRAIN_PATH)
    y_train_full = pd.read_csv(Y_TRAIN_PATH)['target']
    
    print(f"Dimensions des données originales:")
    print(f"X_train: {X_train_full.shape}")
    print(f"y_train: {y_train_full.shape}")
    
    # Diviser les données pour simuler l'environnement d'évaluation
    # Nous utilisons une division par patients pour éviter les fuites
    print("\nDivision des données par patients...")
    
    # Obtenir la liste unique des patients
    unique_patients = X_train_full['patient_id'].unique()
    
    # Diviser les patients en ensembles d'entraînement et de test
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )
    
    print(f"Nombre de patients dans l'entraînement: {len(train_patients)}")
    print(f"Nombre de patients dans le test: {len(test_patients)}")
    
    # Filtrer les données par patients
    train_mask = X_train_full['patient_id'].isin(train_patients)
    test_mask = X_train_full['patient_id'].isin(test_patients)
    
    X_train = X_train_full[train_mask].copy()
    y_train = y_train_full[train_mask].copy()
    X_val = X_train_full[test_mask].copy()
    y_val = y_train_full[test_mask].copy()
    
    print(f"\nDimensions après division:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    
    # Prétraitement des données
    print("\nPrétraitement des données...")
    preprocessor = PreprocessData(X_train, y_train, X_val, y_val)
    
    # Prétraiter les données d'entraînement
    X_train_processed, y_train_processed = preprocessor.process_transformation(is_train=True)
    
    # Prétraiter les données de validation
    X_val_processed, y_val_processed = preprocessor.process_transformation(is_train=False)
    
    # Vérifier s'il reste des valeurs manquantes
    train_na = X_train_processed.isna().sum().sum()
    val_na = X_val_processed.isna().sum().sum()
   
    if train_na > 0:
        print(f"ATTENTION: {train_na} valeurs manquantes dans les données d'entraînement transformées!")
        # Utiliser fillna au lieu de SimpleImputer pour préserver les colonnes
        for col in X_train_processed.columns:
            if X_train_processed[col].isna().any():
                X_train_processed[col] = X_train_processed[col].fillna(X_train_processed[col].median())
    
    if val_na > 0:
        print(f"ATTENTION: {val_na} valeurs manquantes dans les données de validation transformées!")
        # Utiliser fillna au lieu de SimpleImputer pour préserver les colonnes
        for col in X_val_processed.columns:
            if X_val_processed[col].isna().any():
                X_val_processed[col] = X_val_processed[col].fillna(X_train_processed[col].median())
    
    # Vérifier les dimensions après traitement des valeurs manquantes
    print(f"Dimensions après traitement des valeurs manquantes:")
    print(f"X_train_processed: {X_train_processed.shape}")
    print(f"X_val_processed: {X_val_processed.shape}")
    
    # Entraînement d'un modèle
    print("\nEntraînement d'un modèle HistGradientBoosting...")
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X_train_processed, y_train_processed)
    
    # Évaluation sur l'ensemble d'entraînement
    train_preds = model.predict(X_train_processed)
    train_rmse = np.sqrt(mean_squared_error(y_train_processed, train_preds))
    train_r2 = r2_score(y_train_processed, train_preds)
    
    print(f"Performance sur l'entraînement:")
    print(f"  - RMSE: {train_rmse:.2f}")
    print(f"  - R²: {train_r2:.2f}")
    
    # Évaluation sur l'ensemble de validation
    val_preds = model.predict(X_val_processed)
    val_rmse = np.sqrt(mean_squared_error(y_val_processed, val_preds))
    val_r2 = r2_score(y_val_processed, val_preds)
    
    print(f"\nPerformance sur la validation:")
    print(f"  - RMSE: {val_rmse:.2f}")
    print(f"  - R²: {val_r2:.2f}")
    
    # Calculer l'écart entre train et validation
    rmse_diff = val_rmse - train_rmse
    rmse_ratio = val_rmse / train_rmse if train_rmse > 0 else float('inf')
    
    print(f"\nÉcart de RMSE entre validation et entraînement: {rmse_diff:.2f}")
    print(f"Ratio RMSE validation/entraînement: {rmse_ratio:.2f}x")
    
    # Visualisation des prédictions vs réelles
    plt.figure(figsize=(12, 10))
    
    # Distribution des valeurs réelles et prédites
    plt.subplot(2, 2, 1)
    sns.histplot(y_train_processed, color='blue', label='Train réel', alpha=0.6)
    sns.histplot(train_preds, color='red', label='Train prédit', alpha=0.6)
    plt.legend()
    plt.title('Distribution des valeurs (Train)')
    
    plt.subplot(2, 2, 2)
    sns.histplot(y_val_processed, color='blue', label='Validation réel', alpha=0.6)
    sns.histplot(val_preds, color='red', label='Validation prédit', alpha=0.6)
    plt.legend()
    plt.title('Distribution des valeurs (Validation)')
    
    # Scatter plot réel vs prédit
    plt.subplot(2, 2, 3)
    plt.scatter(y_train_processed, train_preds, alpha=0.5)
    plt.plot([min(y_train_processed), max(y_train_processed)], 
             [min(y_train_processed), max(y_train_processed)], 'r--')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.title('Réel vs Prédit (Train)')
    
    plt.subplot(2, 2, 4)
    plt.scatter(y_val_processed, val_preds, alpha=0.5)
    plt.plot([min(y_val_processed), max(y_val_processed)], 
             [min(y_val_processed), max(y_val_processed)], 'r--')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.title('Réel vs Prédit (Validation)')
    
    plt.tight_layout()
    plt.savefig('predictions_analysis.png')
    print("Graphique d'analyse des prédictions sauvegardé.")
    # Analyser la distribution des cibles
    plt.figure(figsize=(10, 6))
    sns.histplot(y_train, bins=50)
    plt.title("Distribution de la variable cible")
    plt.savefig("target_distribution.png")

    # Entraîner un modèle par quantile
    from sklearn.ensemble import GradientBoostingRegressor
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_models = {}

    for q in quantiles:
        model = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=200, max_depth=5
        )
        model.fit(X_train_processed, y_train_processed)
        quantile_models[q] = model
    
    # Analyser les 20 features les plus importantes
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': X_train_processed.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Top 20 Features les plus importantes')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Graphique d'importance des features sauvegardé.")
        
        # Afficher les 10 features les plus importantes
        print("\nTop 10 features les plus importantes:")
        for i, (feature, importance) in enumerate(zip(importance_df['Feature'].head(10), 
                                                    importance_df['Importance'].head(10))):
            print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Si le ratio d'écart est acceptable, entraîner un modèle final
    if rmse_ratio < 1.5:
        print("\nL'écart entre l'entraînement et la validation est acceptable.")
        print("Le modèle semble généraliser correctement aux nouveaux patients.")
    else:
        print("\nATTENTION: L'écart entre l'entraînement et la validation est important.")
        print("Le modèle pourrait avoir des problèmes de généralisation aux nouveaux patients.")
        print("Considérez des ajustements au prétraitement ou au modèle.")

if __name__ == "__main__":
    main()