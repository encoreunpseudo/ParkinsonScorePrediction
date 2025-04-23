import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

'''
fichier pour centraliser toutes les transformations de données

'''
class PreprocessData:
    def __init__(self,path_X,path_y,X_test):
        self.X=path_X
        self.X_test=X_test
        self.X.drop('ledd',axis=1,inplace=True)
        self.X_test.drop('ledd',axis=1,inplace=True)

        self.y = path_y
        self.model_on=None

        self.model_off=None
        self.colonne=None
        self.model_ledd=None
        self.patient_test=self.X_test['patient_id'].tolist()
        self.patient_train=self.X['patient_id']
        

        
        
        
    
    def imputer_valeurs_on_test(self):
        X_copy = self.X_test.copy()[self.colonne.tolist()]
        print(self.colonne)

        X_copy = X_copy.drop(['off','patient_id','time_since_intake_on','time_since_intake_off'],axis=1)

        X_copy['is_missing_on'] = X_copy['on'].isna().astype(int)

        print("Imputations des valeurs de 'on' manquantes sur les données de test...")
        X_copy_a_imputer = X_copy[X_copy['on'].isna()]
        X_copy_connu = X_copy[~X_copy['on'].isna()]
        y_copy = X_copy_connu['on']
        model = self.model_on

        X_copy.loc[X_copy['on'].isna(),'on'] = model.predict(X_copy_a_imputer.drop('on',axis=1))

        self.X_test['on'] = X_copy['on']
        print("Les valeurs de 'on' ont été imputées avec succès.")
        return self.X_test

    def get_data_processed(self):
        return self.X,self.y, self.X_test
        
        
    
    def enlever_index(self,X):
        X.drop('Index',axis=1,inplace=True)
        return X
        
    def encoder_cohort(self,X):

        X['cohort']=X['cohort'].apply(lambda x:1 if x=='A' else 0)
        return X
        
    def encoder_patient(self,X):
        X_patient=X['patient_id']
        vectorizer = CountVectorizer()
        X_patient=vectorizer.fit_transform(X_patient)
        X_patient=pd.DataFrame(X_patient.toarray(),columns=vectorizer.get_feature_names_out())
        X=pd.concat([X,X_patient],axis=1)
        return X
            
        
    def remplir_gene(self,X):
        X_list=X['gene'].tolist()
        def f_l(x):
            if x=='Inconnu':
                return 0.4705
            elif x=='LRRK2+':
                return 1
            else :
                return 0
        def f_g(x):
            if x=='GBA+':
                return 1
            elif x=='Inconnu':
                return 0.4080
            else :
                return 0
        def f_o(x):
            if x=='OTHER+':
                return 1
            elif x=='Inconnu':
                return 0.1211
            else :
                return 0
        for i in range(len(X_list)):
            x=X_list[i]
            
            if type(x)==float:
  
                X_list[i]='Inconnu'
        X['gene']=X_list
        X['est_LRRK2+']=X['gene'].apply(lambda x: f_l(x))
        X['est_GBA+']=X['gene'].apply(lambda x: f_g(x))
        X['est_OTHER+']=X['gene'].apply(lambda x: f_o(x))
        X.drop('gene',axis=1,inplace=True)
        return X
    def virer_patient_et_autre(self,is_test=False):
        if not is_test:
            
            self.X.drop(['time_since_intake_on','time_since_intake_off'],axis=1,inplace=True)
            return self.X
        else:
            
                
            self.X_test.drop(['time_since_intake_on','time_since_intake_off'],axis=1,inplace=True)
            return self.X_test
    
    def rajout_feature_temps(self, X, is_test=False):
        '''
        Ajoute des features temporelles avancées liées à la progression de 'off'
        '''
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression

        X = X.copy()  # pour éviter les effets de bord

        if not is_test:
            X['patient_id'] = self.patient_train
            
        else:
            # rajouter les patients pour les données de test
            X['patient_id'] = self.patient_test


        X['num_visite'] = X.groupby('patient_id').cumcount() + 1
        X['nb_visites'] = X.groupby('patient_id')['num_visite'].transform('max')

        X['diff_on'] = X.groupby('patient_id')['on'].diff().fillna(0)
        X['diff_off'] = X.groupby('patient_id')['off'].diff().fillna(0)

        X['diff_on_first'] = X['on'] - X.groupby('patient_id')['on'].transform('first')
        X['diff_off_first'] = X['off'] - X.groupby('patient_id')['off'].transform('first')

        X['mean_on'] = X.groupby('patient_id')['on'].transform('mean')
        X['mean_off'] = X.groupby('patient_id')['off'].transform('mean')

        X['std_on'] = X.groupby('patient_id')['on'].transform('std').fillna(0)
        X['std_off'] = X.groupby('patient_id')['off'].transform('std').fillna(0)

        X['time_since_last_visit'] = X.groupby('patient_id')['age'].diff().fillna(0)

        X['ratio_on_off'] = X['on'] / (X['off'] + 1e-6)
        X['max_off'] = X.groupby('patient_id')['off'].transform('max')

        # il faut surtout capter les flucutations non linéaires autout de 'off'

        X['moyenne_geometriques_off'] = X.groupby('patient_id')['off'].transform('prod')
        X['moyenne_geometriques_on'] = X.groupby('patient_id')['on'].transform('prod')

        X['relative_diff_off'] = X['diff_off'] / (X['off'] + 1e-6)
        X['relative_diff_on'] = X['diff_on'] / (X['on'] + 1e-6)

        
        X['diff_off_mean'] = X['off'] - X['mean_off']
        X['diff_off_max'] = X['off'] - X['max_off']

        # capturer variabilité autour de la moyenne de off

        X['diff_on_mean'] = X['on'] - X['mean_on']
        X['diff_on_max'] = X['on'] - X['mean_on']




        

        X['ratio_visite']=X['num_visite']/X['nb_visites']
        X['mean_off_prog_interaction']=X['mean_off']*X['ratio_visite']

        X['mean_off_based_on_age']=X.groupby('age')['mean_off'].transform('mean')
        X['mean_off_based_on_disease_duration']=X.groupby('time_since_diagnosis')['mean_off'].transform('mean')



        


        

        return X

    def imputer_age_at_diagnosis(self):
        """
        Impute les valeurs manquantes de age_at_diagnosis en utilisant une régression linéaire.
        """

        
        patients_values = self.X.dropna(subset=['age_at_diagnosis']).groupby('patient_id')['age_at_diagnosis'].first().reset_index()
        patients_values.columns = ['patient_id', 'known_value']
        
        temp_df = self.X.merge(patients_values, on='patient_id', how='left')
        
        mask = temp_df['age_at_diagnosis'].isna() & temp_df['known_value'].notna()
        self.X.loc[mask.values, 'age_at_diagnosis'] = temp_df.loc[mask, 'known_value'].values
        
        patients_sans_diagnostic = self.X.groupby('patient_id')['age_at_diagnosis'].apply(
            lambda x: x.isna().all())
        patients_sans_diagnostic = patients_sans_diagnostic[patients_sans_diagnostic].index.tolist()
        
        nb_patients_sans_diagnostic = len(patients_sans_diagnostic)
        total_patients = len(self.X['patient_id'].unique())
        pourcentage = (nb_patients_sans_diagnostic / total_patients) * 100

        
        if nb_patients_sans_diagnostic > 0:
            patients_avec_diagnostic = ~self.X['patient_id'].isin(patients_sans_diagnostic)
            df_known = self.X[patients_avec_diagnostic].dropna(subset=['age_at_diagnosis'])
            df_known_unique = df_known.drop_duplicates('patient_id')
            
            features = ['age', 'sexM', 'est_LRRK2+', 'est_GBA+', 'est_OTHER+', 'cohort']
            X_known = df_known_unique[features]
            y_known = df_known_unique['age_at_diagnosis']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_known, y_known, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Performance du modèle d'imputation linéaire - MAE: {mae:.2f}, R²: {r2:.2f}")
            
            df_unknown = self.X[self.X['patient_id'].isin(patients_sans_diagnostic)].drop_duplicates('patient_id')
            X_unknown = df_unknown[features]
            predicted_ages = model.predict(X_unknown)
            
            for i, patient_id in enumerate(df_unknown['patient_id']):
                self.X.loc[self.X['patient_id'] == patient_id, 'age_at_diagnosis'] = predicted_ages[i]
        
        missing_pct = self.X['age_at_diagnosis'].isna().mean() * 100
        print(f"Pourcentage de valeurs manquantes après imputation: {missing_pct:.2f}%")
        self.X['time_since_diagnosis'] = self.X['age'] - self.X['age_at_diagnosis']
        print("Variable 'time_since_diagnosis' ajoutée avec succès.")
        
        return self.X
    
    def imputer_globale_on_off(self):
        """
        Impute les valeurs manquantes de 'on' et 'off' en utilisant un modèle entraîné sur l'ensemble train + test.
        Met à jour self.X et self.X_test avec les valeurs imputées.
        """
        from xgboost import XGBRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error

        # Marquer les jeux
        self.X['dataset'] = 'train'
        self.X_test['dataset'] = 'test'

        full_data = pd.concat([self.X, self.X_test], ignore_index=True)
        full_data.reset_index(drop=True, inplace=True)

        print("\n===== IMPUTATION GLOBALE DE 'on' =====")
        features_on = ['age', 'sexM', 'cohort',  'off', 'age_at_diagnosis',
                    'est_LRRK2+', 'est_GBA+', 'est_OTHER+']
        df_known_on = full_data[full_data['on'].notna()].dropna(subset=features_on)
        df_missing_on = full_data[full_data['on'].isna()].copy()

        model_on = RandomForestRegressor(n_estimators=100, random_state=42)
        model_on.fit(df_known_on[features_on], df_known_on['on'])
        full_data.loc[df_missing_on.index, 'on'] = model_on.predict(df_missing_on[features_on])
        self.model_on = model_on

        print("\n===== IMPUTATION GLOBALE DE 'off' =====")
        features_off = ['age', 'sexM', 'cohort',  'on', 'age_at_diagnosis',
                        'est_LRRK2+', 'est_GBA+', 'est_OTHER+']
        df_known_off = full_data[full_data['off'].notna()].dropna(subset=features_off)
        df_missing_off = full_data[full_data['off'].isna()].copy()

        model_off = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, 
                                objective='reg:squarederror', random_state=42, verbosity=0)
        model_off.fit(df_known_off[features_off], df_known_off['off'])
        full_data.loc[df_missing_off.index, 'off'] = model_off.predict(df_missing_off[features_off])
        self.model_off = model_off

        # Remettre les données séparées
        self.X = full_data[full_data['dataset'] == 'train'].drop(columns=['dataset'])
        self.X_test = full_data[full_data['dataset'] == 'test'].drop(columns=['dataset'])

        print("\n✅ Imputation globale terminée.")
        print(self.X.isna().sum())
        return self.X, self.X_test
    def process_transformation(self):

        X_train_copy=self.X.copy()
        X_test_copy=self.X_test.copy()

        X_train_copy=self.encoder_cohort(X_train_copy)
        X_test_copy=self.encoder_cohort(X_test_copy)

        X_train_copy=self.enlever_index(X_train_copy)
        X_test_copy=self.enlever_index(X_test_copy)
        
        X_train_copy=self.remplir_gene(X_train_copy)
        X_test_copy=self.remplir_gene(X_test_copy)


        self.X_test=X_test_copy
        self.X=X_train_copy
        self.X=self.imputer_age_at_diagnosis()
        self.imputer_globale_on_off()
        self.X=self.rajout_feature_temps(self.X)
        self.X_test=self.rajout_feature_temps(self.X_test,is_test=True)

        self.X=self.virer_patient_et_autre()
        self.X_test=self.virer_patient_et_autre(is_test=True)
    def get_data(self):
        return self.X,self.y,self.X_test
    
# on va s'assurer que ca impute bien sur x_test

X_train_full = pd.read_csv('data/X_train_6ZIKlTY.csv')
y_train_full = pd.read_csv('data/y_train_lXj6X5y.csv')['target']
X_test = pd.read_csv('data/X_test_oiZ2ukx.csv')
preprocessor=PreprocessData(X_train_full,y_train_full,X_test)
preprocessor.process_transformation()
X_train,y_train,X_test=preprocessor.get_data()
X_train.to_csv('X_TRainprepro.csv')