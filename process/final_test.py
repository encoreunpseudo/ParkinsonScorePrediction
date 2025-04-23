import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Importer ton préprocesseur
from preprocess_data import PreprocessData

# ----------------------------
# 1. Chargement et prétraitement
# ----------------------------
print("Chargement des données...")
X_train_full = pd.read_csv('data/X_train_6ZIKlTY.csv')
y_train_full = pd.read_csv('data/y_train_lXj6X5y.csv')['target']
X_test = pd.read_csv('data/X_test_oiZ2ukx.csv')

print("Prétraitement des données...")
preprocessor = PreprocessData(X_train_full, y_train_full, X_test)
preprocessor.process_transformation()
X_train, y_train, X_test = preprocessor.get_data()

# Sauvegarder off et patient_id
X_train_off = X_train['off'].copy()
X_test_off = X_test['off'].copy()
train_patient_ids = X_train['patient_id'].copy()
test_patient_ids = X_test['patient_id'].copy()

# ----------------------------
# 2. Feature Engineering centré sur les progressions temporelles
# ----------------------------
def create_progression_features(X, patient_ids, is_test=False):
    """
    Crée des features avancées centrées sur la progression temporelle,
    en se basant sur l'importance observée de diff_off et des features de déviation.
    """
    print(f"Création des features de progression temporelle pour {'test' if is_test else 'train'}...")
    
    # Créer un DataFrame pour stocker toutes les features
    result_df = pd.DataFrame()
    
    # Ajouter patient_id et features de base
    result_df['patient_id'] = patient_ids
    for col in X.columns:
        result_df[col] = X[col].values
    
    # Créer un dictionnaire pour stocker les groupes par patient
    patient_groups = {}
    
    for patient_id in result_df['patient_id'].unique():
        # Sélectionner les données du patient
        mask = result_df['patient_id'] == patient_id
        patient_data = result_df[mask].copy()
        
        # S'assurer que les données sont triées par âge
        if 'age' in patient_data:
            patient_data = patient_data.sort_values('age')
        elif 'num_visite' in patient_data:
            patient_data = patient_data.sort_values('num_visite')
        
        # Stocker dans le dictionnaire
        patient_groups[patient_id] = patient_data
    
    # Initialiser un DataFrame vide pour stocker les résultats
    enhanced_df = pd.DataFrame()
    
    # Pour chaque patient, calculer les features de progression
    for patient_id, patient_data in patient_groups.items():
        # Créer des features de progression et de déviation
        
        # 1. Features de différence
        if len(patient_data) > 1:
            patient_data['diff_off_previous'] = patient_data['off'].diff()
            patient_data['diff_off_previous'].iloc[0] = 0  # Premier point
            
            # Différence relative
            patient_data['diff_off_relative'] = patient_data['diff_off_previous'] / (patient_data['off'].shift(1) + 1e-5)
            patient_data['diff_off_relative'].iloc[0] = 0
            
            # Différence cumulée depuis la première visite
            patient_data['diff_off_cumulative'] = patient_data['off'] - patient_data['off'].iloc[0]
            
            # Accélération (changement de différence)
            patient_data['off_acceleration'] = patient_data['diff_off_previous'].diff()
            patient_data['off_acceleration'].iloc[0:2] = 0
            
            # Variation de la tendance
            patient_data['trend_change'] = (patient_data['diff_off_previous'] > 0).astype(int).diff()
            patient_data['trend_change'].iloc[0:2] = 0
        else:
            # Pour les patients avec une seule visite
            patient_data['diff_off_previous'] = 0
            patient_data['diff_off_relative'] = 0
            patient_data['diff_off_cumulative'] = 0
            patient_data['off_acceleration'] = 0
            patient_data['trend_change'] = 0
        
        # 2. Features de position dans la trajectoire
        if len(patient_data) > 1:
            off_range = patient_data['off'].max() - patient_data['off'].min()
            if off_range > 0:
                patient_data['trajectory_position'] = (patient_data['off'] - patient_data['off'].min()) / off_range
            else:
                patient_data['trajectory_position'] = 0.5
        else:
            patient_data['trajectory_position'] = 0.5
        
        # 3. Features de déviation
        if 'mean_off' in patient_data.columns:
            # Utiliser la moyenne existante si disponible
            patient_data['off_deviation'] = patient_data['off'] - patient_data['mean_off']
            patient_data['off_rel_deviation'] = patient_data['off_deviation'] / (patient_data['mean_off'] + 1e-5)
            
            # Également calculer une rolling mean pour d'autres features
            patient_data['rolling_mean'] = patient_data['mean_off']
        else:
            # Calculer la moyenne mobile si pas disponible
            if len(patient_data) > 1:
                patient_data['rolling_mean'] = patient_data['off'].expanding().mean()
                patient_data['off_deviation'] = patient_data['off'] - patient_data['rolling_mean']
                patient_data['off_rel_deviation'] = patient_data['off_deviation'] / (patient_data['rolling_mean'] + 1e-5)
            else:
                patient_data['rolling_mean'] = patient_data['off']
                patient_data['off_deviation'] = 0
                patient_data['off_rel_deviation'] = 0
        
        # 4. Volatilité et stabilité
        if len(patient_data) > 2:
            # Volatilité (écart-type mobile)
            patient_data['rolling_std'] = patient_data['off'].expanding().std()
            patient_data['volatility'] = patient_data['rolling_std'] / (patient_data['rolling_mean'] + 1e-5)
            
            # Stabilité (basée sur les changements relatifs)
            if len(patient_data) > 1:
                abs_rel_changes = np.abs(patient_data['diff_off_relative'].iloc[1:]).mean()
                patient_data['stability'] = 1 / (1 + abs_rel_changes)
            else:
                patient_data['stability'] = 1
        else:
            patient_data['rolling_std'] = 0
            patient_data['volatility'] = 0
            patient_data['stability'] = 1
        
        # 5. Position temporelle
        if 'num_visite' in patient_data.columns and 'nb_visites' in patient_data.columns:
            patient_data['visit_progress'] = patient_data['num_visite'] / patient_data['nb_visites']
        else:
            # Si num_visite n'est pas disponible, créer une mesure similaire
            total_rows = len(patient_data)
            patient_data['visit_progress'] = [(i+1)/total_rows for i in range(total_rows)]
        
        # 6. Interactions importantes
        patient_data['off_deviation_x_visit_progress'] = patient_data['off_deviation'] * patient_data['visit_progress']
        if 'time_since_diagnosis' in patient_data.columns:
            patient_data['off_x_time_since_diagnosis'] = patient_data['off'] * patient_data['time_since_diagnosis']
            patient_data['off_deviation_x_time_since_diagnosis'] = patient_data['off_deviation'] * patient_data['time_since_diagnosis']
        
        # 7. Features avancées de changement
        if len(patient_data) > 2:
            # Extrapoler la tendance
            try:
                X = np.array(range(len(patient_data))).reshape(-1, 1)
                y = patient_data['off'].values
                from sklearn.linear_model import LinearRegression
                model = LinearRegression().fit(X, y)
                patient_data['off_trend_slope'] = model.coef_[0]
                patient_data['off_trend_residual'] = patient_data['off'] - model.predict(X)
            except:
                patient_data['off_trend_slope'] = 0
                patient_data['off_trend_residual'] = 0
        else:
            patient_data['off_trend_slope'] = 0
            patient_data['off_trend_residual'] = 0
        
        # Ajouter les données traitées au DataFrame final
        enhanced_df = pd.concat([enhanced_df, patient_data])
    
    # Rétablir l'ordre original des lignes si nécessaire
    if not is_test:
        enhanced_df = enhanced_df.sort_index()
    
    print(f"Features de progression créées. Dimension: {enhanced_df.shape}")
    return enhanced_df

# Appliquer l'ingénierie de features
X_train_enhanced = create_progression_features(X_train, train_patient_ids)
X_test_enhanced = create_progression_features(X_test, test_patient_ids, is_test=True)

# Supprimer patient_id pour l'entraînement
if 'patient_id' in X_train_enhanced.columns:
    X_train_enhanced = X_train_enhanced.drop('patient_id', axis=1)
if 'patient_id' in X_test_enhanced.columns:
    X_test_enhanced = X_test_enhanced.drop('patient_id', axis=1)

# ----------------------------
# 3. Sélection des features les plus pertinentes
# ----------------------------
def select_best_features(X, y, threshold=0.95):
    """Sélectionne les features les plus importantes."""
    print("\nSélection des features les plus importantes...")
    
    # Utiliser XGBoost pour évaluer l'importance des features
    selector = XGBRegressor(n_estimators=100, random_state=42)
    selector.fit(X, y)
    
    # Obtenir les importances
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Calculer l'importance cumulative
    importance_df['cumulative'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()
    
    # Sélectionner les features qui contribuent à threshold% de l'importance
    selected_features = importance_df[importance_df['cumulative'] <= threshold]['feature'].tolist()
    
    print(f"Features sélectionnées: {len(selected_features)}/{X.shape[1]}")
    
    # Afficher les top features
    top_n = min(20, len(selected_features))
    print(f"Top {top_n} features:")
    for i, feature in enumerate(selected_features[:top_n], 1):
        importance = importance_df[importance_df['feature'] == feature]['importance'].values[0]
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Visualiser l'importance
    plt.figure(figsize=(12, 8))
    plt.bar(range(top_n), importance_df['importance'].head(top_n))
    plt.xticks(range(top_n), importance_df['feature'].head(top_n), rotation=90)
    plt.title('Importance des features')
    plt.tight_layout()
    plt.savefig('progression_feature_importance.png')
    print("Graphique d'importance sauvegardé dans 'progression_feature_importance.png'")
    
    return selected_features, importance_df

# Calculer le bruit comme dans ton approche originale
print("Calcul du bruit (différence entre target et off)...")
noise_values = y_train - X_train_off

# Sélectionner les meilleures features
selected_features, importance_df = select_best_features(X_train_enhanced, noise_values)
X_train_selected = X_train_enhanced[selected_features]
X_test_selected = X_test_enhanced[selected_features]

# ----------------------------
# 4. Entraînement du modèle et validation
# ----------------------------
def train_progression_model(X, y, X_test):
    """Entraîne un modèle optimisé pour les features de progression temporelle."""
    print("\nEntraînement du modèle de progression...")
    
    # Définir le modèle LightGBM avec hyperparamètres optimisés
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=8,
        num_leaves=127,
        min_child_samples=10,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    
    # Validation croisée
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    print("Validation croisée:")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        cv_scores.append(rmse)
        
        print(f"  Fold {fold+1}: RMSE = {rmse:.4f}")
    
    mean_rmse = np.mean(cv_scores)
    std_rmse = np.std(cv_scores)
    print(f"RMSE moyen: {mean_rmse:.4f} ± {std_rmse:.4f}")
    
    # Entraîner sur toutes les données
    print("Entraînement du modèle final sur toutes les données...")
    model.fit(X, y)
    
    # Prédire sur les données de test
    noise_pred = model.predict(X_test)
    
    return model, noise_pred

# Entraîner le modèle et obtenir les prédictions
model, noise_pred = train_progression_model(X_train_selected, noise_values, X_test_selected)

# ----------------------------
# 5. Prédiction finale et soumission
# ----------------------------
print("\nCréation des prédictions finales...")
final_pred = X_test_off + noise_pred

print("Statistiques des prédictions:")
print(f"  Moyenne: {np.mean(final_pred):.4f}")
print(f"  Écart-type: {np.std(final_pred):.4f}")
print(f"  Min: {np.min(final_pred):.4f}")
print(f"  Max: {np.max(final_pred):.4f}")

# Création du fichier de soumission
print("\nCréation du fichier de soumission...")
submission = pd.DataFrame({
    'ID': range(len(final_pred)),
    'target': final_pred
})

# sauvegarder l 'importance des features et les metrics clés en png
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.bar(range(len(importance_df)), importance_df['importance'])
plt.xticks(range(len(importance_df)), importance_df['feature'], rotation=90)
plt.title('Importance des features')
plt.tight_layout()
plt.savefig('progression_feature_importance.png')

# Sauvegarder le fichier
submission.to_csv('submission_progression.csv', index=False)

# Sauvegarder le modèle
import pickle
with open('progression_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'selected_features': selected_features
    }, f)

print("\nProcessus terminé avec succès!")
print("Fichier de soumission créé: 'submission_progression.csv'")
print("Modèle sauvegardé: 'progression_model.pkl'")