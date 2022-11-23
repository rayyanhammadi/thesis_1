import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler

# CLassification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier

class Models:
    def __init__(self, df, date_split: int, step_ahead: int, nb_years_lag = 0):
        self.df = df
        self.date_split = date_split
        self.nb_years_lag = nb_years_lag
        self.step_ahead = step_ahead
        self.range_data_split = range(self.date_split, len(df))
        self.X = df.iloc[:, :-1] # la matrice X explicatives
        self.Y = df.iloc[:,-1] # variable à expliquer
        #self.Y_test_label = pd.concat([self.Y, pd.DataFrame(np.nan, index=self.df.index, columns=['RF_label'])], axis=1)
        #self.Y_test_probs = pd.concat([self.Y, pd.DataFrame(np.nan, index=self.df.index, columns=['RF_probs'])], axis=1)
        self.Y_train_US = None
        self.Y_test_US = None
        self.X_train_US = None
        self.X_test_US = None
        self.var_imp_RF = pd.DataFrame(np.nan, index=df.index, columns=self.X.columns)
        self.var_imp_GB = pd.DataFrame(np.nan, index=df.index, columns=self.X.columns)
        self.RF_model = None
        #self.RF = RandomForestClassifier()
        #self.BG = BaggingClassifier()
        #self.GBC = GradientBoostingClassifier()

    def split_1(self):
        for id_split in self.range_data_split:

            if self.nb_years_lag == 0:
                self.Y_train_US = self.df.iloc[0:id_split - self.step_ahead + 1, -1]
                self.X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]
            else:
                self.Y_train_US = self.df.iloc[id_split - 12 * self.nb_years_lag:id_split - self.step_ahead + 1, -1]
                self.X_train_US = self.X.iloc[id_split - 12 * self.nb_years_lag:id_split - self.step_ahead + 1, :]    

            self.Y_test_US = self.df.iloc[id_split:, -1]
            self.X_test_US = self.X.iloc[id_split:, :]
        


    def split_2(self):

        #todo à tester la fonction: train_test_split()
        self.X_train_US, self.X_test_US, self.Y_train_US, self.Y_test_US = train_test_split(self.X, self.Y, train_size = self.date_split, shuffle=False)


    def fit(self):
        self.RF_model = RandomForestClassifier(n_estimators=2000,random_state=42).fit(self.X_train_US,self.Y_train_US)
    def predict(self):
        self.Y_test_US_1  = pd.DataFrame(self.Y_test_US, columns=['USA (Acc_Slow)'])
        self.Y_test_US_1["RF_labels"] = self.RF_model.predict(self.X_test_US)
        self.Y_test_US_1["RF_probs"] = self.RF_model.predict_proba(self.X_test_US)[:,1]
        #result_label = self.RF_model.predict(self.X_test_US)
        #result_probs = self.Y_test_US["RF_probas"] = self.RF_model.predict_proba(self.X_test_US)
    def plot(self):
        y_label = self.Y_test_US_1.iloc[:,0] # variable témoin
        y_label_RF = self.Y_test_US_1.iloc[:,1] #prédiction du modéle
        y_probs_RF = self.Y_test_US_1.iloc[:,-1] # prédiction probas
        plt.plot(y_probs_RF)
        fig, ax = plt.subplots()
        ax.plot(y_probs_RF.index, y_probs_RF, color='black')
        threshold = 0.5
        ax.axhline(threshold, color='gray', lw=2, alpha=0.7)
        ax.fill_between(y_label.index, 0, 1, where=y_label == 1,
                        color='gray', alpha=0.5, transform=ax.get_xaxis_transform())
        plt.show()

    def show_matrix_confusion(self):
        pass
        
#if __name__ == '__main__':
#   model_instance = Models()
#    model_instance.split(0.2)
#    model_instance.fit()
#    print(model_instance.predict())
#    print("Accuracy: ",     model_instance.model.score(model_instance.X_test, model_instance.y_test))
