import pandas as pd
import numpy as np
import math

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler

# CLassification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier

class Models:
    def __init__(self, df):
        self.logistic_reg = LogisticRegression()
        self.RF = RandomForestClassifier()
        self.BG = BaggingClassifier()
        self.GBC = GradientBoostingClassifier()

    def split(self, test_size):
        X = np.array(self.df[['Humidity', 'Pressure (millibars)']])
        y = np.array(self.df['Temperature (C)'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    def fit(self):
        self.model = self.linear_reg.fit(self.X_train, self.y_train)
    def predict(self):
        result = self.linear_reg.predict(self.X_test)
        return result
if __name__ == '__main__':
    model_instance = Model()
    model_instance.split(0.2)
    model_instance.fit()
    print(model_instance.predict())
    print("Accuracy: ",     model_instance.model.score(model_instance.X_test, model_instance.y_test))


