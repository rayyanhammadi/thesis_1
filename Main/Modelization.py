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

#Performance
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

#Plotly
pd.options.plotting.backend = "plotly"
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.figure_factory import create_table

class Models:
    def __init__(self, Y, X, date_split: int, step_ahead: int, nb_years_lag = 0):
        self.date_split = date_split
        self.nb_years_lag = nb_years_lag
        self.step_ahead = step_ahead
        self.X = X
        self.Y = Y
        #self.Y_test_probs_label = pd.concat([self.Y, pd.DataFrame(np.nan, index=self.df.index, columns=['RF_label'])], axis=1)
        #self.Y_test_probs_probs = pd.concat([self.Y, pd.DataFrame(np.nan, index=self.df.index, columns=['RF_probs'])], axis=1)
        self.Y_train_US = None
        self.Y_test_probs = pd.concat(
            [self.Y, pd.DataFrame(np.nan, index=self.Y.index, columns=['RF_probs'])],
            axis=1)
        self.Y_test_label = pd.concat(
            [self.Y, pd.DataFrame(np.nan, index=self.Y.index, columns=['RF_label'])],
            axis=1)
        self.X_train_US = None
        self.X_test = None
        self.var_imp = pd.DataFrame(np.nan, index=X.index, columns=self.X.columns)


    def plot(self):
        y_label = self.Y.iloc[204:]
        # y_label_RF = self.Y_test_probs_US_1.iloc[:,1] #prédiction du modéle
        y_probs_RF = self.Y_test_probs.iloc[:,-1] # prédiction probas
        plt.plot(y_probs_RF)
        fig, ax = plt.subplots()
        ax.plot(y_probs_RF.index, y_probs_RF, color='black')
        threshold = 0.5
        ax.axhline(threshold, color='gray', lw=2, alpha=0.7)
        ax.fill_between(y_label.index, 0, 1, where=y_label == 1,
                        color='gray', alpha=0.5, transform=ax.get_xaxis_transform())
        plt.show()

    def show_confusion_matrix(self):

        y = self.Y_test_label.dropna()
        labels = y.iloc[:,0]
        preds = y.iloc[:,-1]
        confusion_df = pd.DataFrame(confusion_matrix(y_true=labels,y_pred=preds),
                                    index=["Slowdown", "Acceleration"],
                                    columns=["Predicted slowdown", "Predicted Acceleration"])

        fig = create_table(confusion_df, index=True)

        fig.update_layout(autosize=False, width=450, height=75)

        fig.show()
        pass
    def predict(self):
        self.expanding_window_prediction_method()
        aggregate_var_imp_RF_v1 = pd.DataFrame(np.nansum(self.var_imp, axis=0).T, index=self.X.columns, columns=['Importance'])
        aggregate_var_imp_RF_v1.index.name = 'variables'
        print(aggregate_var_imp_RF_v1.sort_values(by="Importance", ascending=False).head(10))

    def expanding_window_prediction_method(self):
        print("1 step-ahead training and predicting with expanding window method..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:

            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
            X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

            X_test_US = self.X.iloc[id_split:, :]

            model_forest = RandomForestClassifier(n_estimators=2000, random_state=42).fit(X_train_US, Y_train_US)
            self.var_imp.iloc[id_split, :] = model_forest.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split,-1] = model_forest.predict_proba(X_test_US)[0][1]
            self.Y_test_label.iloc[id_split, -1] = model_forest.predict(X_test_US)[0]
    def rolling_window_prediction_method(self,window_lenghth=18):
        print("1 step-ahead training and predicting with rolling window method..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:

            print("predicting  Y_%i | X[%i:%i,:]" % (id_split,id_split - self.step_ahead - window_lenghth, id_split - self.step_ahead + 1 ))
            Y_train_US = self.Y.iloc[id_split - self.step_ahead - window_lenghth:id_split - self.step_ahead + 1]
            X_train_US = self.X.iloc[id_split - self.step_ahead - window_lenghth:id_split - self.step_ahead + 1, :]

            X_test_US = self.X.iloc[id_split:, :]

            model_forest = RandomForestClassifier(n_estimators=2000, random_state=42).fit(X_train_US, Y_train_US)
            self.var_imp.iloc[id_split, :] = model_forest.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split,-1] = model_forest.predict_proba(X_test_US)[0][1]

#if __name__ == '__main__':
#   model_instance = Models()
#    model_instance.split(0.2)
#    model_instance.fit()
#    print(model_instance.predict())
#    print("Accuracy: ",     model_instance.model.score(model_instance.X_test, model_instance.y_test))
