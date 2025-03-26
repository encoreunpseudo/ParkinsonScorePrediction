import pandas as pd
import numpy as np
'''
fichier pour centraliser toutes les transformations de données

'''
class PreprocessData:
    def __init__(self,path_X,path_y):
        self.X = pd.read_csv(path_X)
        self.y = pd.read_csv(path_y)
        print("coucou")
        self.virer_patient()
        print("coucou 2")
        self.remplir_gene()
        print("Coucou 3")
        
        
    
    def valeurs_off(self):
        pass

    def remplir_gene(self):
        pass

    def virer_patient(self):
        self.X.drop('patient_id',axis=1,inplace=True)
        
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
        valeurs=['LRRK2+','No Mutation','GBA+','OTHER+','Inconnu']
        self.X['est_LRRK2+']=self.X['gene'].apply(lambda x: f_l(x))
        self.X['est_GBA+']=self.X['gene'].apply(lambda x: f_g(x))
        self.X['est_OTHER+']=self.X['gene'].apply(lambda x: f_o(x))
        self.X.drop('gene',axis=1,inplace=True)
        return self.X
    def get_X(self):
        return self.X

        
preprocess4 = PreprocessData('data/X_train_6ZIKlTY.csv', 'data/y_train_lXj6X5y.csv')
print(preprocess4.get_X().head(30))