import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
class PreprocessData:
    def __init__(self, Xt, yt=None, Xv=None, yv=None):
        """
        Initialise le préprocesseur avec les données d'entraînement et de validation
        
        Args:
            Xt (DataFrame): Données d'entraînement X
            yt (Series/DataFrame): Labels d'entraînement y
            Xv (DataFrame): Données de validation/test X
            yv (Series/DataFrame, optional): Labels de validation/test y
        """
        self.X_train = Xt
        self.y_train = yt
        self.X_val = Xv
        self.y_val = yv

        self.patient_train= self.X_train['patient_id']
        self.patient_val= self.X_val['patient_id']
        
        # Pour stocker les modèles d'imputation
        self.age_diagnosis_model = None
        self.on_imputation_model = None
        
        # Pour stocker les vectorizers et autres transformateurs
        self.patient_vectorizer = None
        
        # Pour stocker l'ordre des colonnes
        self.feature_order = None
        self.age_diagnosis_features = None
        self.on_feature_cols = None

    def process_transformation(self, is_train=True):
        """
        Méthode principale qui applique toutes les transformations
        
        Args:
            is_train (bool): True si on traite les données d'entraînement, False sinon
        
        Returns:
            tuple: (X transformé, y)
        """
        if is_train==True:
            X = self.X_train.copy()
            y = self.y_train
        else:
            X = self.X_val.copy()
            y = self.y_val
            
        # Étapes de transformation communes
        X = self.enlever_index(X)
        X = self.remplir_gene(X)
        X = self.encoder_cohort(X)
        
        # Imputation des valeurs "off" (doit être fait avant "on" pour profiter de la relation)
        if is_train:
            X = self.imputer_valeurs_off_train(X)
        else:
            X = self.imputer_valeurs_off_val(X)
        
        # Imputation de l'âge au diagnostic - différent selon train/test
        if is_train:
            X = self.imputer_age_at_diagnosis_train(X)
        else:
            # Vérifier si time_since_diagnosis existe déjà dans les données de validation
            if 'time_since_diagnosis' in X.columns:
                print("La colonne 'time_since_diagnosis' existe déjà dans les données de validation.")
            else:
                X = self.imputer_age_at_diagnosis_val(X)
        
        # Vectorisation des patients
        X = self.encoder_patient(X, is_train)
        
        # Imputation des valeurs "on"
        if is_train:
            X = self.imputer_valeurs_on_train(X)
            X=self.rajout_feature_temps(X)
        else:
            X = self.imputer_valeurs_on_val(X)
            X=self.rajout_feature_temps(X,is_test=True)
            
        # Autres transformations
        
        X = self.virer_patient_et_autre(X)
        
        if is_train:
            # Sauvegarde de l'ordre des colonnes après toutes les transformations
            self.feature_order = X.columns.tolist()
            self.X_train = X
            return X, y
        else:
            # Vérifier si les colonnes dans les données de test sont différentes
            missing_cols = [col for col in self.feature_order if col not in X.columns]
            extra_cols = [col for col in X.columns if col not in self.feature_order]
            
            if missing_cols:
                print(f"Attention : colonnes manquantes dans les données de test : {missing_cols}")
                # Créer des colonnes vides pour les colonnes manquantes
                for col in missing_cols:
                    X[col] = 0
                    
            if extra_cols:
                print(f"Attention : colonnes supplémentaires dans les données de test : {extra_cols}")
                # Supprimer les colonnes supplémentaires
                X = X.drop(columns=extra_cols)
            
            # Réorganiser les colonnes dans le même ordre que pour l'entraînement
            X = X[self.feature_order]
            
            self.X_val = X
            return X, y
            
    def enlever_index(self, X):
        """Supprime la colonne Index"""
        if 'Index' in X.columns:
            X = X.drop('Index', axis=1)
        return X
        
    def encoder_cohort(self, X):
        """Encode la colonne cohort en binaire"""
        X['cohort'] = X['cohort'].apply(lambda x: 1 if x=='A' else 0)
        return X
        
    def encoder_patient(self, X, is_train=True):
        """
        Encode les IDs patients avec CountVectorizer
        
        Args:
            X (DataFrame): Les données à transformer
            is_train (bool): Si True, on fit le vectorizer, sinon on réutilise
        
        Returns:
            DataFrame: Les données avec les colonnes encodées
        """
        X_patient = X['patient_id']
        
        if is_train:
            self.patient_vectorizer = CountVectorizer()
            X_patient_encoded = self.patient_vectorizer.fit_transform(X_patient)
        else:
            # Pour les données de test, utiliser seulement les colonnes vues pendant l'entraînement
            # et gérer les nouveaux patients inconnus
            X_patient_encoded = self.patient_vectorizer.transform(X_patient)
            
        X_patient_df = pd.DataFrame(
            X_patient_encoded.toarray(),
            columns=self.patient_vectorizer.get_feature_names_out(),
            index=X.index
        )
        
        return pd.concat([X, X_patient_df], axis=1)
            
    def remplir_gene(self, X):
        """Remplit les valeurs manquantes dans la colonne gene et crée des features binaires"""
        X_list = X['gene'].tolist()
        
        def f_l(x):
            if x=='Inconnu':
                return 0.4705
            elif x=='LRRK2+':
                return 1
            else:
                return 0
                
        def f_g(x):
            if x=='GBA+':
                return 1
            elif x=='Inconnu':
                return 0.4080
            else:
                return 0
                
        def f_o(x):
            if x=='OTHER+':
                return 1
            elif x=='Inconnu':
                return 0.1211
            else:
                return 0
                
        for i in range(len(X_list)):
            x = X_list[i]
            if type(x) == float:
                X_list[i] = 'Inconnu'
                
        X['gene'] = X_list
        X['est_LRRK2+'] = X['gene'].apply(lambda x: f_l(x))
        X['est_GBA+'] = X['gene'].apply(lambda x: f_g(x))
        X['est_OTHER+'] = X['gene'].apply(lambda x: f_o(x))
        X.drop('gene', axis=1, inplace=True)
        
        return X

    def virer_patient_et_autre(self, X):
        """Supprime les colonnes qui ne sont plus nécessaires"""
        cols_to_drop = ['patient_id']
        
        if 'time_since_intake_on' in X.columns:
            cols_to_drop.append('time_since_intake_on')
        
        if 'time_since_intake_off' in X.columns:
            cols_to_drop.append('time_since_intake_off')
            
        if 'disease_duration' in X.columns and 'time_since_diagnosis' not in X.columns:
            X['time_since_diagnosis'] = X['disease_duration']
            cols_to_drop.append('disease_duration')
            
        X = X.drop(cols_to_drop, axis=1)
        return X

    def rajout_feature_temps(self, X,is_test=False):
        """Ajoute des features temporelles basées sur les groupes de patients"""
        X_copy = X.copy()
        # rajout des patients
        if is_test:
            X_copy['patient_id'] = self.patient_val
            print(X_copy.isna().sum())
        else:    
            X_copy['patient_id'] = self.patient_train
        
        X_copy['num_visite'] = X_copy.groupby('patient_id').cumcount() + 1

        X_copy['nb_visites'] = X_copy.groupby('patient_id')['num_visite'].transform('max')

        X_copy['diff_on'] = X_copy.groupby('patient_id')['on'].diff()
        
        mask_first_visit = X_copy['diff_on'].isna()
        X_copy.loc[mask_first_visit, 'diff_on'] = 0

        X_copy['diff_on_first'] = X_copy.groupby('patient_id')['on'].transform('first')
        X_copy['diff_on_first'] = X_copy['on'] - X_copy['diff_on_first']

        X_copy['mean_on'] = X_copy.groupby('patient_id')['on'].transform('mean')

        X_copy['std_on'] = X_copy.groupby('patient_id')['on'].transform('std')
        X_copy['std_on'] = X_copy['std_on'].fillna(0)

        X_copy['time_since_last_visit'] = X_copy.groupby('patient_id')['age'].diff()
        
        X_copy.loc[mask_first_visit, 'time_since_last_visit'] = 0
        
        return X_copy

    def imputer_age_at_diagnosis_train(self, X):
        """
        Impute les valeurs manquantes de age_at_diagnosis sur les données d'entraînement
        et entraîne un modèle pour les données de validation.
        """
        patients_values = X.dropna(subset=['age_at_diagnosis']).groupby('patient_id')['age_at_diagnosis'].first().reset_index()
        patients_values.columns = ['patient_id', 'known_value']
        
        temp_df = X.merge(patients_values, on='patient_id', how='left')
        
        mask = temp_df['age_at_diagnosis'].isna() & temp_df['known_value'].notna()
        X.loc[mask.values, 'age_at_diagnosis'] = temp_df.loc[mask, 'known_value'].values
        
        patients_sans_diagnostic = X.groupby('patient_id')['age_at_diagnosis'].apply(
            lambda x: x.isna().all())
        patients_sans_diagnostic = patients_sans_diagnostic[patients_sans_diagnostic].index.tolist()
        
        nb_patients_sans_diagnostic = len(patients_sans_diagnostic)
        total_patients = len(X['patient_id'].unique())
        pourcentage = (nb_patients_sans_diagnostic / total_patients) * 100
        
        self.age_diagnosis_features = ['age', 'sexM', 'est_LRRK2+', 'est_GBA+', 'est_OTHER+', 'cohort']
        
        if nb_patients_sans_diagnostic > 0:
            patients_avec_diagnostic = ~X['patient_id'].isin(patients_sans_diagnostic)
            df_known = X[patients_avec_diagnostic].dropna(subset=['age_at_diagnosis'])
            df_known_unique = df_known.drop_duplicates('patient_id')
            
            X_known = df_known_unique[self.age_diagnosis_features]
            y_known = df_known_unique['age_at_diagnosis']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_known, y_known, test_size=0.2, random_state=42)
            
            self.age_diagnosis_model = LinearRegression()
            self.age_diagnosis_model.fit(X_train, y_train)
            
            y_pred = self.age_diagnosis_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Performance du modèle d'imputation age_at_diagnosis - MAE: {mae:.2f}, R²: {r2:.2f}")
            
            df_unknown = X[X['patient_id'].isin(patients_sans_diagnostic)].drop_duplicates('patient_id')
            X_unknown = df_unknown[self.age_diagnosis_features]
            predicted_ages = self.age_diagnosis_model.predict(X_unknown)
            
            for i, patient_id in enumerate(df_unknown['patient_id']):
                X.loc[X['patient_id'] == patient_id, 'age_at_diagnosis'] = predicted_ages[i]
        
        missing_pct = X['age_at_diagnosis'].isna().mean() * 100
        print(f"Pourcentage de valeurs manquantes après imputation: {missing_pct:.2f}%")
        X['time_since_diagnosis'] = X['age'] - X['age_at_diagnosis']
        print("Variable 'time_since_diagnosis' ajoutée avec succès.")
        
        return X
        
    def imputer_age_at_diagnosis_val(self, X):
        """
        Impute les valeurs manquantes de age_at_diagnosis sur les données de validation
        en utilisant le modèle entraîné précédemment.
        """
        if 'time_since_diagnosis' in X.columns:
            print("La colonne 'time_since_diagnosis' existe déjà dans les données de validation. Conservation des valeurs existantes.")
            return X
            
        if self.age_diagnosis_model is None:
            raise ValueError("Le modèle d'imputation d'âge n'a pas été entraîné. Appelez d'abord process_transformation sur les données d'entraînement.")
        
        if not hasattr(self, 'age_diagnosis_features'):
            raise ValueError("Les features pour le modèle d'âge n'ont pas été enregistrées.")
        
        patients_values = X.dropna(subset=['age_at_diagnosis']).groupby('patient_id')['age_at_diagnosis'].first().reset_index()
        patients_values.columns = ['patient_id', 'known_value']
        
        temp_df = X.merge(patients_values, on='patient_id', how='left')
        
        mask = temp_df['age_at_diagnosis'].isna() & temp_df['known_value'].notna()
        X.loc[mask.values, 'age_at_diagnosis'] = temp_df.loc[mask, 'known_value'].values
        
        mask_missing = X['age_at_diagnosis'].isna()
        if mask_missing.any():
            missing_cols = [col for col in self.age_diagnosis_features if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes dans les données de validation pour age_at_diagnosis: {missing_cols}")
                
            X_missing = X.loc[mask_missing, self.age_diagnosis_features]
            
            predicted_ages = self.age_diagnosis_model.predict(X_missing)
            X.loc[mask_missing, 'age_at_diagnosis'] = predicted_ages
            
        if 'time_since_diagnosis' not in X.columns:
            X['time_since_diagnosis'] = X['age'] - X['age_at_diagnosis']
            print("Variable 'time_since_diagnosis' ajoutée aux données de validation.")
        
        return X
        
    def imputer_valeurs_on_train(self, X):
        """
    Impute les valeurs manquantes de 'on' avec des méthodes avancées
    
    Args:
        X (DataFrame): Les données à transformer
        
    Returns:
        DataFrame: Les données avec 'on' imputé
    """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.impute import KNNImputer
        
        X_copy = X.copy()
        
        cols_to_exclude = ['on']
        if 'ledd' in X_copy.columns:
            cols_to_exclude.append('ledd')
        if 'off' in X_copy.columns:
            cols_to_exclude.append('off')
        if 'time_since_intake_on' in X_copy.columns:
            cols_to_exclude.append('time_since_intake_on')
        if 'time_since_intake_off' in X_copy.columns:
            cols_to_exclude.append('time_since_intake_off')
        if 'patient_id' in X_copy.columns:
            cols_to_exclude.append('patient_id')
            
        mask_missing = X_copy['on'].isna()
        
        if not mask_missing.any():
            print("Aucune valeur 'on' à imputer.")
            return X
        
        print("Imputation préliminaire des autres features...")
        features_with_missing = X_copy.columns[X_copy.isna().any()].tolist()
        features_with_missing = [col for col in features_with_missing if col != 'on'] 
        
        if features_with_missing:
            preliminary_imputer = KNNImputer(n_neighbors=5)
            X_copy[features_with_missing] = preliminary_imputer.fit_transform(X_copy[features_with_missing])
        
        print("Ajout de features supplémentaires pour l'imputation...")
        
        if 'patient_id' in X_copy.columns:
            patient_stats = X_copy.groupby('patient_id')['on'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
            patient_stats.columns = ['patient_id', 'patient_mean_on', 'patient_median_on', 'patient_min_on', 'patient_max_on', 'patient_std_on']
            patient_stats.fillna(0, inplace=True)  # Pour les patients avec une seule observation
            
            X_copy = X_copy.merge(patient_stats, on='patient_id', how='left')
        
        feature_cols = [col for col in X_copy.columns if col not in cols_to_exclude]
        self.on_feature_cols = feature_cols
        
        X_with_on = X_copy[~mask_missing]
        X_missing_on = X_copy[mask_missing]
        
        X_train_on = X_with_on[feature_cols]
        y_train_on = X_with_on['on']
        
        print(f"Entraînement du modèle d'imputation pour {sum(mask_missing)} valeurs manquantes de 'on'...")
        
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        self.on_imputation_model = gb_model
        self.on_imputation_model.fit(X_train_on, y_train_on)
        
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, mean_absolute_error
        
        mae_scorer = make_scorer(mean_absolute_error)
        cv_scores = cross_val_score(gb_model, X_train_on, y_train_on, cv=5, scoring=mae_scorer)
        
        print(f"Performance du modèle d'imputation de 'on' - MAE CV: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
        
        if mask_missing.any():
            X_missing_features = X_missing_on[feature_cols]
            predicted_values = self.on_imputation_model.predict(X_missing_features)
            X.loc[mask_missing, 'on'] = predicted_values
            
        return X
    
    def imputer_valeurs_on_val(self, X):
        """
        Impute les valeurs manquantes de 'on' sur les données de validation
        en utilisant le modèle entraîné précédemment.
        """
        if self.on_imputation_model is None:
            raise ValueError("Le modèle d'imputation de 'on' n'a pas été entraîné. Appelez d'abord process_transformation sur les données d'entraînement.")
        
        if not hasattr(self, 'on_feature_cols'):
            raise ValueError("Les colonnes de features pour le modèle 'on' n'ont pas été enregistrées.")
            
        mask_missing = X['on'].isna()
        
        if not mask_missing.any():
            print("Aucune valeur 'on' à imputer dans les données de validation.")
            return X
            
        print(f"Imputation de {sum(mask_missing)} valeurs 'on' dans les données de validation...")
        
        X_copy = X.copy()
        
        features_with_missing = X_copy.columns[X_copy.isna().any()].tolist()
        features_with_missing = [col for col in features_with_missing if col != 'on' and col in self.on_feature_cols]
        
        if features_with_missing:
            from sklearn.impute import KNNImputer
            preliminary_imputer = KNNImputer(n_neighbors=5)
            X_copy[features_with_missing] = preliminary_imputer.fit_transform(X_copy[features_with_missing])
        
        if 'patient_id' in X_copy.columns:
            patient_stats = X_copy.groupby('patient_id')['on'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
            patient_stats.columns = ['patient_id', 'patient_mean_on', 'patient_median_on', 'patient_min_on', 'patient_max_on', 'patient_std_on']
            patient_stats.fillna(0, inplace=True)
            
            X_copy = X_copy.merge(patient_stats, on='patient_id', how='left')
        
        missing_cols = [col for col in self.on_feature_cols if col not in X_copy.columns]
        
        for col in missing_cols:
            print(f"Ajout de la colonne manquante '{col}' avec des zéros.")
            X_copy[col] = 0
        
        X_missing = X_copy[mask_missing]
        X_missing_features = X_missing[self.on_feature_cols]
        
        predicted_values = self.on_imputation_model.predict(X_missing_features)
        X.loc[mask_missing, 'on'] = predicted_values
        
        return X
    def imputer_valeurs_off_train(self, X):
        """
    Impute les valeurs manquantes de 'off' avec des méthodes avancées
    
    Args:
        X (DataFrame): Les données à transformer
        
    Returns:
        DataFrame: Les données avec 'off' imputé
    """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.impute import KNNImputer
        
        # Vérifier si la colonne 'off' existe
        if 'off' not in X.columns:
            print("La colonne 'off' n'existe pas dans les données. Aucune imputation effectuée.")
            return X
        
        # Copie pour éviter les warnings
        X_copy = X.copy()
        
        # Colonnes à exclure pour la prédiction
        cols_to_exclude = ['off']
        if 'ledd' in X_copy.columns:
            cols_to_exclude.append('ledd')
        if 'on' in X_copy.columns:
            cols_to_exclude.append('on')  # on exclut 'on' pour éviter la fuite d'information
        if 'time_since_intake_on' in X_copy.columns:
            cols_to_exclude.append('time_since_intake_on')
        if 'time_since_intake_off' in X_copy.columns:
            cols_to_exclude.append('time_since_intake_off')
        if 'patient_id' in X_copy.columns:
            cols_to_exclude.append('patient_id')
            
        # Séparation des données avec/sans valeurs 'off'
        mask_missing = X_copy['off'].isna()
        
        # Si aucune valeur manquante, retourner directement
        if not mask_missing.any():
            print("Aucune valeur 'off' à imputer.")
            return X
        
        # Nombre de valeurs manquantes et pourcentage
        n_missing = mask_missing.sum()
        percent_missing = (n_missing / len(X_copy)) * 100
        print(f"Imputation de {n_missing} valeurs manquantes ({percent_missing:.2f}%) pour 'off'...")
        
        # 1. Imputation préliminaire avec KNN pour les autres caractéristiques qui pourraient être manquantes
        features_with_missing = X_copy.columns[X_copy.isna().any()].tolist()
        features_with_missing = [col for col in features_with_missing if col != 'off' and col != 'on']  # Exclure off et on
        
        if features_with_missing:
            print(f"Imputation préliminaire des features: {features_with_missing}")
            preliminary_imputer = KNNImputer(n_neighbors=5)
            feature_data = X_copy[features_with_missing]
            X_copy[features_with_missing] = preliminary_imputer.fit_transform(feature_data)
        
        # 2. Créer des features additionnelles pour améliorer l'imputation
        
        # 2.1 Si on a des données longitudinales (plusieurs visites par patient)
        if 'patient_id' in X_copy.columns:
            # Calculer des statistiques par patient pour 'off' (pour les patients qui ont des valeurs)
            patient_stats = X_copy.groupby('patient_id')['off'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
            patient_stats.columns = ['patient_id', 'patient_mean_off', 'patient_median_off', 'patient_min_off', 'patient_max_off', 'patient_std_off']
            patient_stats.fillna(0, inplace=True)  # Pour les patients avec une seule ou aucune observation
            
            # Si on a aussi des valeurs pour 'on', on peut utiliser la relation on/off
            if 'on' in X_copy.columns:
                # Patients avec des valeurs pour 'on' et 'off'
                valid_mask = ~X_copy['on'].isna() & ~X_copy['off'].isna()
                if valid_mask.sum() > 0:
                    # Rapport moyen on/off par patient
                    patient_on_off = X_copy[valid_mask].groupby('patient_id').apply(
                        lambda x: x['on'].mean() / x['off'].mean() if x['off'].mean() != 0 else 1
                    ).reset_index()
                    patient_on_off.columns = ['patient_id', 'patient_on_off_ratio']
                    
                    # Fusionner avec les autres statistiques
                    patient_stats = patient_stats.merge(patient_on_off, on='patient_id', how='left')
                    patient_stats['patient_on_off_ratio'].fillna(1, inplace=True)  # Valeur par défaut
            
            # Fusionner avec nos données principales
            X_copy = X_copy.merge(patient_stats, on='patient_id', how='left')
            
            # Pour les patients sans statistiques, utiliser les valeurs globales
            for col in patient_stats.columns:
                if col != 'patient_id' and col in X_copy.columns:
                    mask = X_copy[col].isna()
                    if mask.any() and col.startswith('patient_mean'):
                        X_copy.loc[mask, col] = X_copy.loc[~mask, col].mean()
                    elif mask.any() and col.startswith('patient_median'):
                        X_copy.loc[mask, col] = X_copy.loc[~mask, col].median()
                    elif mask.any():
                        X_copy.loc[mask, col] = 0
        
        # 2.2 Utiliser la relation avec 'on' quand disponible
        if 'on' in X_copy.columns:
            global_ratio = 1.0  # Valeur par défaut
            valid_mask = ~X_copy['on'].isna() & ~X_copy['off'].isna()
            
            if valid_mask.sum() > 0:
                # Moyenne globale du rapport on/off
                global_on = X_copy.loc[valid_mask, 'on'].mean()
                global_off = X_copy.loc[valid_mask, 'off'].mean()
                if global_off != 0:
                    global_ratio = global_on / global_off
                
                # Créer une feature prédictive utilisant on et le ratio
                X_copy['predicted_off_from_on'] = X_copy['on'] / global_ratio
                X_copy['predicted_off_from_on'].fillna(X_copy['off'].mean(), inplace=True)
        
        # Sélectionner toutes les colonnes sauf celles à exclure pour le modèle
        feature_cols = [col for col in X_copy.columns if col not in cols_to_exclude]
        
        # Stocker ces colonnes pour garantir le même ordre lors de la prédiction
        self.off_feature_cols = feature_cols
        
        X_with_off = X_copy[~mask_missing]
        X_missing_off = X_copy[mask_missing]
        
        # Extraire les données d'entraînement 
        X_train_off = X_with_off[feature_cols]
        y_train_off = X_with_off['off']
        
        # 3. Utiliser un modèle plus robuste
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,  # Un peu plus profond pour capturer des relations plus complexes
            subsample=0.8,  # Ajouter du sous-échantillonnage pour éviter le surapprentissage
            random_state=42
        )
        
        # Entraîner le modèle
        self.off_imputation_model = gb_model
        self.off_imputation_model.fit(X_train_off, y_train_off)
        
        # Évaluer la performance sur l'ensemble d'entraînement
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        train_preds = self.off_imputation_model.predict(X_train_off)
        
        train_mae = mean_absolute_error(y_train_off, train_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train_off, train_preds))
        train_r2 = r2_score(y_train_off, train_preds)
        
        print(f"Performance du modèle d'imputation 'off' sur l'entraînement:")
        print(f"  - MAE: {train_mae:.2f}")
        print(f"  - RMSE: {train_rmse:.2f}")
        print(f"  - R²: {train_r2:.2f}")
        
        # Évaluer avec validation croisée également
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer
        
        cv_mae = -np.mean(cross_val_score(gb_model, X_train_off, y_train_off, cv=3, 
                                        scoring=make_scorer(mean_absolute_error)))
        
        print(f"  - MAE CV (3-fold): {cv_mae:.2f}")
        
        # Appliquer le modèle aux données manquantes
        if mask_missing.any():
            X_missing_features = X_missing_off[feature_cols]
            predicted_values = self.off_imputation_model.predict(X_missing_features)
            X.loc[mask_missing, 'off'] = predicted_values
            
        return X

    def imputer_valeurs_off_val(self, X):
        """
    Impute les valeurs manquantes de 'off' sur les données de validation
    en utilisant le modèle entraîné précédemment.
    """
        # Vérifier si la colonne 'off' existe
        if 'off' not in X.columns:
            print("La colonne 'off' n'existe pas dans les données de validation. Aucune imputation effectuée.")
            return X
            
        if not hasattr(self, 'off_imputation_model'):
            raise ValueError("Le modèle d'imputation de 'off' n'a pas été entraîné. Appelez d'abord process_transformation sur les données d'entraînement.")
        
        if not hasattr(self, 'off_feature_cols'):
            raise ValueError("Les colonnes de features pour le modèle 'off' n'ont pas été enregistrées.")
            
        # Identification des lignes avec valeurs manquantes
        mask_missing = X['off'].isna()
        
        if not mask_missing.any():
            print("Aucune valeur 'off' à imputer dans les données de validation.")
            return X
            
        print(f"Imputation de {sum(mask_missing)} valeurs 'off' dans les données de validation...")
        
        # Préparation des données de validation (similaire à l'entraînement)
        X_copy = X.copy()
        
        # 1. Imputation préliminaire des autres features
        features_with_missing = X_copy.columns[X_copy.isna().any()].tolist()
        features_with_missing = [col for col in features_with_missing if col != 'off' and col != 'on' and col in self.off_feature_cols]
        
        if features_with_missing:
            from sklearn.impute import KNNImputer
            preliminary_imputer = KNNImputer(n_neighbors=5)
            X_copy[features_with_missing] = preliminary_imputer.fit_transform(X_copy[features_with_missing])
        
        # 2. Recréer les features additionnelles comme à l'entraînement
        
        # 2.1 Statistiques par patient
        if 'patient_id' in X_copy.columns:
            patient_stats = X_copy.groupby('patient_id')['off'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
            patient_stats.columns = ['patient_id', 'patient_mean_off', 'patient_median_off', 'patient_min_off', 'patient_max_off', 'patient_std_off']
            patient_stats.fillna(0, inplace=True)
            
            # Ratio on/off si possible
            if 'on' in X_copy.columns:
                valid_mask = ~X_copy['on'].isna() & ~X_copy['off'].isna()
                if valid_mask.sum() > 0:
                    patient_on_off = X_copy[valid_mask].groupby('patient_id').apply(
                        lambda x: x['on'].mean() / x['off'].mean() if x['off'].mean() != 0 else 1
                    ).reset_index()
                    patient_on_off.columns = ['patient_id', 'patient_on_off_ratio']
                    
                    patient_stats = patient_stats.merge(patient_on_off, on='patient_id', how='left')
                    patient_stats['patient_on_off_ratio'].fillna(1, inplace=True)
            
            X_copy = X_copy.merge(patient_stats, on='patient_id', how='left')
            
            # Imputer des valeurs manquantes dans les statistiques
            for col in patient_stats.columns:
                if col != 'patient_id' and col in X_copy.columns:
                    mask = X_copy[col].isna()
                    if mask.any() and col.startswith('patient_mean'):
                        X_copy.loc[mask, col] = X_copy.loc[~mask, col].mean() if (~mask).any() else 0
                    elif mask.any() and col.startswith('patient_median'):
                        X_copy.loc[mask, col] = X_copy.loc[~mask, col].median() if (~mask).any() else 0
                    elif mask.any():
                        X_copy.loc[mask, col] = 0
        
        # 2.2 Relation avec 'on'
        if 'on' in X_copy.columns:
            # Utiliser le même ratio global qu'à l'entraînement si possible
            # Si non, estimer à partir des données actuelles
            global_ratio = 1.0
            valid_mask = ~X_copy['on'].isna() & ~X_copy['off'].isna()
            
            if valid_mask.sum() > 0:
                global_on = X_copy.loc[valid_mask, 'on'].mean()
                global_off = X_copy.loc[valid_mask, 'off'].mean()
                if global_off != 0:
                    global_ratio = global_on / global_off
            
            X_copy['predicted_off_from_on'] = X_copy['on'] / global_ratio
            X_copy['predicted_off_from_on'].fillna(X_copy['off'].mean() if ~X_copy['off'].isna().all() else 0, inplace=True)
        
        # Vérifier les colonnes manquantes par rapport à l'entraînement
        missing_cols = [col for col in self.off_feature_cols if col not in X_copy.columns]
        
        # Ajouter les colonnes manquantes avec des valeurs par défaut
        for col in missing_cols:
            print(f"Ajout de la colonne manquante '{col}' avec des zéros.")
            X_copy[col] = 0
        
        # Sélectionner les features dans le même ordre que lors de l'entraînement
        X_missing = X_copy[mask_missing]
        try:
            X_missing_features = X_missing[self.off_feature_cols]
        except KeyError as e:
            print(f"Erreur lors de la sélection des features: {e}")
            print(f"Colonnes manquantes: {[col for col in self.off_feature_cols if col not in X_missing.columns]}")
            raise
        
        # Prédiction et imputation
        predicted_values = self.off_imputation_model.predict(X_missing_features)
        X.loc[mask_missing, 'off'] = predicted_values
        
        return X
    def get_X_train(self):
        return self.X_train
        
    def get_y_train(self):
        return self.y_train
        
    def get_X_val(self):
        return self.X_val
        
    def get_y_val(self):
        return self.y_val
        
    def get_train_data(self):
        return self.X_train, self.y_train
        
    def get_val_data(self):
        return self.X_val, self.y_val

