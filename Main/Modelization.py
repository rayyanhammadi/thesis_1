from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from Main.Data_processing import Data #on importe la class Data préalablement définie dans Data_processing

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler

# CLassification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier

class Models:
    def __init__(self, df, date_split: int, step_ahead: int, nb_years_lag = 0):
        self.df = df
        self.date_split = date_split    #on prendra plus tard date_split=204, c'est le nombre de variables trainset, de couleur jaune et vert sur l'excel
        self.nb_years_lag = nb_years_lag    # For a recursive window on the training sample (usually 15 years), and 0 for an expanding window
        self.step_ahead = step_ahead    # From 1 month to 12 months
        self.range_data_split = range(self.date_split, len(df)) #variable qui prend les données du test set rouge dans l'excel
        self.X = df.iloc[:, :-1] # la matrice X des variables explicatives (comme en MLG) cad notre excel entier privé de la dernière colonne, la colonne de Y
        self.Y = df.iloc[:,-1] # la matrice Y des variables binaires à expliquer cad la dernière colonne USA (Acc_Slow) de notre excel   
        self.Y_test_label = pd.concat([self.Y, pd.DataFrame(np.nan, index=self.df.index, columns=['RF_label'])], axis=1)
        self.Y_test_probs = pd.concat([self.Y, pd.DataFrame(np.nan, index=self.df.index, columns=['RF_probs'])], axis=1)
        self.Y_train_US = None
        self.Y_test_US = None
        self.X_train_US = None
        self.X_test_US = None
        self.var_imp_RF = pd.DataFrame(np.nan, index=df.index, columns=self.X.columns)
        self.var_imp_GB = pd.DataFrame(np.nan, index=df.index, columns=self.X.columns)
        self.RF_model = None
        self.RF = RandomForestClassifier()
        self.BG = BaggingClassifier()
        self.GBC = GradientBoostingClassifier()

    def split_1(self):    #donne deux parties qui sont les données train set pour entraîner le modele et les traintest pour predire
        for id_split in self.range_data_split:

            if self.nb_years_lag == 0:
                self.Y_train_US = self.df.iloc[0:id_split - self.step_ahead + 1, -1]
                self.X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]
            else:
                self.Y_train_US = self.df.iloc[id_split - 12 * self.nb_years_lag:id_split - self.step_ahead + 1, -1]
                self.X_train_US = self.X.iloc[id_split - 12 * self.nb_years_lag:id_split - self.step_ahead + 1, :]    

            self.Y_test_US = self.df.iloc[id_split:, -1]
            self.X_test_US = self.X.iloc[id_split:, :]
        


    def split_2(self):  #fais la même chose que split_1 mais utilise train_test_split qui est deja codée sur python. On fait ca pour verifier que split_1 est bien

        #todo à tester la fonction: train_test_split()
        self.X_train_US, self.X_test_US, self.Y_train_US, self.Y_test_US = train_test_split(self.X, self.Y, train_size = self.date_split, shuffle=False)


    def fit(self):   #  predict label for the next obs only, entraine notre modele de RF sur le trainset qui est self.X_train_US et self.Y_train_US 
        self.RF_model = RandomForestClassifier(n_estimators=2000,random_state=42).fit(self.X_train_US,self.Y_train_US) #ça crée 2000 arbres de décision à partir d'un sous-ensemble de l'ensemble d'apprentissage sélectionné de manière aléatoire.

    def predict(self):  # predict probabilities for the next obs only and keep prediction for class 1 only, cad donne Ŷt, estimation de Yt obtenu de la fonction fit ci-dessus
        self.Y_test_US_1  = pd.DataFrame(self.Y_test_US, columns=['USA (Acc_Slow)'])
        self.Y_test_US_1["RF_labels"] = self.RF_model.predict(self.X_test_US)
        self.Y_test_US_1["RF_probs"] = self.RF_model.predict_proba(self.X_test_US)[:,1]
        result_label = self.RF_model.predict(self.X_test_US)
        result_probs = self.Y_test_US["RF_probas"] = self.RF_model.predict_proba(self.X_test_US)

#je rajoute du code à partir d'ici
    def plot1(self): #affiche des graphiques des predictions du modele en utilisant le code du prof
        self.y_label = self.y_test_label_v1.iloc[self.date_split:,0]
        self.y_label_RF = self.y_test_label_v1.iloc[self.date_split:,1]
        self.y_label_GB = self.y_test_label_v1.iloc[self.date_split:,2]
        self.y_probs_RF = self.y_test_probs_v1.iloc[self.date_split:,1]
        self.y_probs_GB = self.y_test_probs_v1.iloc[self.date_split:,2]
        plt.plot(self.y_probs_RF)  #nous donne la courbe RF noire similaire à la diapo 42. C'est la simulation de Ŷt par la méthode de RF avec t variant de 2002 à 2020, cad t varie dans notre dataset (de couleur rouge sur l'excel)










#if __name__ == '__main__':
#   model_instance = Models()
#    model_instance.split(0.2)
#    model_instance.fit()
#    print(model_instance.predict())
#    print("Accuracy: ", model_instance.model.score(model_instance.X_test, model_instance.y_test))
