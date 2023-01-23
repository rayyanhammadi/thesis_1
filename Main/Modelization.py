import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import seaborn as sns


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler

# CLassification
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier

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
    def __init__(self,name:str, Y, X, date_split: int, step_ahead: int, nb_years_lag = 0):
        self.name = name
        self.date_split = date_split
        self.nb_years_lag = nb_years_lag
        self.step_ahead = step_ahead
        self.X = X
        self.Y = Y
        #self.Y_test_probs_label = pd.concat([self.Y, pd.DataFrame(np.nan, index=self.df.index, columns=['RF_label'])], axis=1)
        #self.Y_test_probs_probs = pd.concat([self.Y, pd.DataFrame(np.nan, index=self.df.index, columns=['RF_probs'])], axis=1)
        self.Y_train_US = None
        self.Y_test_probs = pd.concat(
            [self.Y, pd.DataFrame(np.nan, index=self.Y.index, columns=["LGT_probs"])],
            axis=1)
        self.Y_test_label = pd.concat(
            [self.Y, pd.DataFrame(np.nan, index=self.Y.index, columns=['LGT_label'])],
            axis=1)
        self.X_train_US = None
        self.X_test = None
        self.var_imp = pd.DataFrame(np.nan, index=X.index, columns=self.X.columns)


    def plot(self):
        y_label = self.Y.iloc[204:]
        # y_label_RF = self.Y_test_probs_US_1.iloc[:,1] #prédiction du modéle
        y_probs_RF = self.Y_test_probs.iloc[:,-1] # prédiction probas
        # plt.plot(y_probs_RF)
        fig, ax = plt.subplots()
        ax.plot(y_probs_RF.index, y_probs_RF, color='black')
        threshold = 0.5
        ax.axhline(threshold, color='gray', lw=2, alpha=0.7)
        ax.fill_between(y_label.index, 0, 1, where=y_label == 1,
                        color='gray', alpha=0.5, transform=ax.get_xaxis_transform())
        plt.title(str(self.name))
        plt.show()
    @staticmethod
    def make_confusion_matrix(cf,
                              group_names=None,
                              categories='auto',
                              count=True,
                              percent=True,
                              cbar=True,
                              xyticks=True,
                              xyplotlabels=True,
                              sum_stats=True,
                              figsize=None,
                              cmap='Blues',
                              title=None):
        '''
        This function will make a pretty plot of a sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Arguments
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html

        title:         Title for the heatmap. Default is None.
        '''

        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names) == cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace(cf) / float(np.sum(cf))

            # if it is a binary confusion matrix, show some more stats
            if len(cf) == 2:
                # Metrics for Binary Confusion Matrices
                precision = cf[1, 1] / sum(cf[:, 1])
                recall = cf[1, 1] / sum(cf[1, :])
                f1_score = 2 * precision * recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy, precision, recall, f1_score)
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize == None:
            # Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks == False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title)
        plt.show()
    def show_confusion_matrix(self):

        y = self.Y_test_label.dropna()
        labels = y.iloc[:,0]
        preds = y.iloc[:,-1]
        confusion_df = confusion_matrix(labels,preds)
        labels=["True Neg","False Pos","False Neg","True Pos"]
        categories = ["Slowdown", "Acceleration"]
        self.make_confusion_matrix(cf = confusion_df,group_names=labels,categories=categories,cmap='binary',)

    def predict(self):
        if self.name == "logit":
            self.logit_model()
        elif self.name == "RF":
            self.RF_model()
        elif self.name == "EN":
            self.EN_model()
        elif self.name == "ABC":
            self.ABC_model()
        aggregate_var_imp_RF_v1 = pd.DataFrame(np.nansum(self.var_imp, axis=0).T, index=self.X.columns, columns=['Importance'])
        aggregate_var_imp_RF_v1.index.name = 'variables'
        print(aggregate_var_imp_RF_v1.sort_values(by="Importance", ascending=False).head(10))


    def logit_model(self):
        print(str(self.step_ahead) + " step-ahead training and predicting with logit model..")
        cols= ['spread63m', 'spread13m', 'spread23m',
       'spread53m', 'spread103m', 'spread102','spread105','spread52','spread21']
        X = self.X
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:

            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
            X_train_US = X.iloc[0:id_split - self.step_ahead + 1, :]

            X_test_US = X.iloc[id_split:, :]

            logit = LogisticRegression(max_iter=1000).fit(X_train_US, Y_train_US)
            # self.var_imp.iloc[id_split, :] = logit.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split,-1] = logit.predict_proba(X_test_US)[0][1]
            self.Y_test_label.iloc[id_split,-1] = logit.predict(X_test_US)[0]
    def RF_model(self):
        print(str(self.step_ahead) + " tep-ahead training and predicting with expanding window method..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:

            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
            X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

            X_test_US = self.X.iloc[id_split:, :]

            model_forest = RandomForestClassifier(n_estimators=2000, random_state=42).fit(X_train_US, Y_train_US)
            self.var_imp.iloc[id_split, :] = model_forest.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split,1] = model_forest.predict_proba(X_test_US)[0][1]
            self.Y_test_label.iloc[id_split, 1] = model_forest.predict(X_test_US)[0]

    def EN_model(self):
        print(str(self.step_ahead) + " tep-ahead training and predicting with expanding window method..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:

            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
            X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

            X_test_US = self.X.iloc[id_split:, :]

            en = ElasticNet().fit(X_train_US, Y_train_US)
            # self.var_imp.iloc[id_split, :] = en.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split,1] = en.predict(X_test_US)[0]
            self.Y_test_label.iloc[id_split, 1] = en.predict(X_test_US)[0]
    def ABC_model(self):
        print(str(self.step_ahead) + " tep-ahead training and predicting with expanding window method..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:

            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
            X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

            X_test_US = self.X.iloc[id_split:, :]

            abc = AdaBoostClassifier(n_estimators=2000).fit(X_train_US, Y_train_US)
            self.var_imp.iloc[id_split, :] = abc.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split,1] = abc.predict(X_test_US)[0]
            self.Y_test_label.iloc[id_split, 1] = abc.predict(X_test_US)[0]

#if __name__ == '__main__':
#   model_instance = Models()
#    model_instance.split(0.2)
#    model_instance.fit()
#    print(model_instance.predict())
#    print("Accuracy: ",     model_instance.model.score(model_instance.X_test, model_instance.y_test))
