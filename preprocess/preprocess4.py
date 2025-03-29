import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
'''
fichier pour centraliser toutes les transformations de données

'''
class PreprocessData:
    def __init__(self,path_X,path_y):
        self.X=path_X

        self.y = path_y
  

        self.enlever_index()
        self.remplir_gene()
        #self.virer_patient()
        self.encoder_cohort()
        self.imputer_age_at_diagnosis()
        #self.rajout_feature_temps()
        self.encoder_patient()
        
        
    
    def enlever_index(self):
        self.X.drop('Index',axis=1,inplace=True)
        
    def encoder_cohort(self):
        self.X['cohort']=self.X['cohort'].apply(lambda x:1 if x=='A' else 0)
        
    def encoder_patient(self):
        X_patient=self.X['patient_id']
        vectorizer = CountVectorizer()
        X_patient=vectorizer.fit_transform(X_patient)
        X_patient=pd.DataFrame(X_patient.toarray(),columns=vectorizer.get_feature_names_out())
        self.X=pd.concat([self.X,X_patient],axis=1)
        self.X.drop('patient_id',axis=1,inplace=True)
        return self.X
            
        
    def remplir_gene(self):
        X_list=self.X['gene'].tolist()
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
        self.X['gene']=X_list
        self.X['est_LRRK2+']=self.X['gene'].apply(lambda x: f_l(x))
        self.X['est_GBA+']=self.X['gene'].apply(lambda x: f_g(x))
        self.X['est_OTHER+']=self.X['gene'].apply(lambda x: f_o(x))
        self.X.drop('gene',axis=1,inplace=True)
        return self.X
    def virer_patient(self):
        self.X.drop('patient_id',axis=1,inplace=True)
    def get_X(self):
        return self.X
    def get_y(self):
        return self.y
    def get_data(self):
        return self.X,self.y
    def rajout_feature_temps(self):
        '''
        Pour capturer la relation temporelle
        '''
        
        # rajouter le numéro de la visite
        self.X['num_visite'] = self.X.groupby('patient_id').cumcount() + 1

        # rajouter le nombre de visite total
        self.X['nb_visites'] = self.X.groupby('patient_id')['num_visite'].transform('max')

        # rajouter la progression du score on et off depuis la dernière visite
        self.X['diff_on'] = self.X.groupby('patient_id')['on'].diff()
        self.X['diff_off'] = self.X.groupby('patient_id')['off'].diff()

        # rajouter la progression du score on et off depuis la première visite
        self.X['diff_on_first'] = self.X.groupby('patient_id')['on'].transform('first')
        self.X['diff_off_first'] = self.X.groupby('patient_id')['off'].transform('first')

        # rajouter la moyenne du score on et off sur toutes les visites
        self.X['mean_on'] = self.X.groupby('patient_id')['on'].transform('mean')
        self.X['mean_off'] = self.X.groupby('patient_id')['off'].transform('mean')

        # rajouter l'écart type du score on et off sur toutes les visites
        self.X['std_on'] = self.X.groupby('patient_id')['on'].transform('std')
        self.X['std_off'] = self.X.groupby('patient_id')['off'].transform('std')

        # rajouter le temps depuis la dernière visite
        self.X['time_since_last_visit'] = self.X.groupby('patient_id')['age'].diff()

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
        self.X['disease_duration'] = self.X['age'] - self.X['age_at_diagnosis']
        print("Variable 'disease_duration' ajoutée avec succès.")
        
        return self.X
        