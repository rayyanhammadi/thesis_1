import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from statistics import mean
from collections import Counter
from scipy.stats import bernoulli
import seaborn as sns
from sklearn.decomposition import PCA


# CLassification
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
#Performance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,\
    roc_auc_score, roc_curve, confusion_matrix ,mean_squared_error

# Cross validation
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

class Models:
    @staticmethod
    def rolling_PCA(data, important_features, n_comp=.99):
        """
        Effectue une PCA sur la matrice X

        :param data:
        :param important_features:
        :param n_comp:
        :return:
        """
        pca = PCA(n_components=n_comp)
        pca.fit_transform(data)
        n_pcs = pca.n_components_
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        initial_feature_names = data.columns
        most_important_features = [*set([initial_feature_names[most_important[i]] for i in range(n_pcs)])]
        if important_features:
            print(most_important_features)
        return most_important_features
    def __init__(self,name:str, Y, X, date_split: int, step_ahead: int):
        """
        Initialisation de l'objet "models" et des variables qui seront utiles à la prédiction

        :param name: Nom du modèle (cf. la fonction predict())
        :param Y: La variable à prédire
        :param X: La matrice des covariables
        :param date_split: La date à partir de laquelle on prédit les Y
        :param step_ahead: Le pas
        """

        self.name = name
        self.date_split = date_split
        self.step_ahead = step_ahead
        self.X = X
        self.Y = Y
        self.Y_test_probs = pd.concat(
            [self.Y, pd.DataFrame(np.nan, index=self.Y.index, columns=["probs"])],
            axis=1)
        self.Y_test_label = pd.concat(
            [self.Y, pd.DataFrame(np.nan, index=self.Y.index, columns=['label'])],
            axis=1)
        self.var_imp = pd.DataFrame(np.nan, index=X.index, columns=self.X.columns)


    def plot(self):
        """
        Trace les prédictions du modèle
        :return:
        """

        y_label = self.Y.iloc[self.date_split:]
        y_probs = self.Y_test_probs.iloc[:,-1]
        # plt.plot(y_probs_RF)
        fig, ax = plt.subplots()
        ax.plot(y_probs.index, y_probs, color='black')
        threshold = 0.5 #Seuil par défaut
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
                              title=None,labels=None,preds=None):
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
            # if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            mse = mean_squared_error(labels,preds)
            ras = roc_auc_score(labels,preds)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nMse={:0.3f}" \
                         "\n ROC_AUC Score={:0.3f}".format(

                accuracy, precision, recall, f1_score,mse,ras)
        #     else:
        #         stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        # else:
        #     stats_text = ""

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
        """

        Renvoie la matrice de confusion du modéle

        """
        labels = self.Y.iloc[self.date_split:]
        preds = self.Y_test_label.iloc[:,-1].dropna()

        confusion_df = confusion_matrix(labels,preds)
        label=["True Neg","False Pos","False Neg","True Pos"]
        categories = ["Slowdown", "Acceleration"]
        self.make_confusion_matrix(cf = confusion_df,group_names=label,categories=categories,cmap='binary',labels=labels,preds=preds)


    def predict(self):

        if self.name == "logit":
            self.logit_model()
        elif self.name == "RF":
            self.RF_model()
        elif self.name == "EN":
            self.EN_model()
        elif self.name == "ABC":
            self.ABC_model()
        elif self.name == "CV_RF":
            self.CV_RF_model()
        elif self.name == "GB":
            self.GB_model()
        elif self.name == "BC":
            self.BC_model()
        elif self.name == "DTR":
            self.DTR_model()
        elif self.name=="KNN":
            self.KNN_model()

        aggregate_var_imp_RF_v1 = pd.DataFrame(np.nansum(self.var_imp, axis=0).T, index=self.X.columns, columns=['Importance'])
        aggregate_var_imp_RF_v1.index.name = 'variables'
        print(aggregate_var_imp_RF_v1.sort_values(by="Importance", ascending=False).head(10))


    def logit_model(self,method_1=False
                    ,method_2=True,threshold_tuning=False):
        print(str(self.step_ahead) + " step-ahead training and predicting with logit model..")



        range_data_split = range(self.date_split, len(self.X))
        opt_thresholds=np.zeros((len(range_data_split)))
        naif = False
        for id_split in range_data_split:
            if method_1:
                print("predicting  Y_%i | X[0:%i,:] with method 1" % (id_split, id_split - self.step_ahead + 1 ))
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
                Y_hat_US_labs, Y_hat_US_probs = Y_train_US.copy(), Y_train_US.copy()
                if id_split<431:
                    for i in range(self.step_ahead):
                        X_train_US = self.X.iloc[0:id_split - self.step_ahead + i + 1, :]
                        X_test_US = self.X.iloc[id_split - self.step_ahead + i + 2:, :]
                        Y_train_tilde = Y_hat_US_labs.iloc[0:id_split - self.step_ahead + i + 1]

                        model = LogisticRegression(max_iter=5000).fit(X_train_US,Y_train_tilde)
                        Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = model.predict_proba(X_test_US)[0][1]
                        Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = model.predict(X_test_US)[0]


                if id_split < 431:
                    self.Y_test_probs.iloc[id_split, -1] = Y_hat_US_probs[id_split]
                    self.Y_test_label.iloc[id_split,-1] = Y_hat_US_labs[id_split]
            elif method_2:
                print("predicting  Y_%i | X[0:%i,:] with method 2" % (id_split, id_split - self.step_ahead + 1))
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
                Y_train_US_hat = Y_train_US.copy()
                if naif:
                    for i in range(self.step_ahead-1):
                        Y_train_US_hat[self.Y.index[id_split - self.step_ahead + i + 2]] = Y_train_US[id_split - self.step_ahead]

                    X_train_US = self.X.iloc[0:id_split, :]

                    X_test_US = self.X.iloc[id_split:, :]

                    model = LogisticRegression(max_iter=1000).fit(X_train_US, Y_train_US_hat)
                    self.Y_test_probs.iloc[id_split, -1] = model.predict_proba(X_test_US)[0][1]
                    self.Y_test_label.iloc[id_split, -1] = model.predict(X_test_US)[0]
                else:

                    if Counter(Y_train_US[id_split - self.step_ahead - 3:id_split]).most_common()[0][0]==1:
                        print(Counter(Y_train_US[id_split - self.step_ahead - 6:id_split]).most_common()[0])
                        p_hat = mean(Y_train_US[id_split - self.step_ahead - 6:id_split])
                    else:
                        p_hat = mean(Y_train_US[id_split - self.step_ahead - 6:id_split])
                    print(p_hat)
                    for i in range(self.step_ahead-1):
                        Y_train_US_hat[self.Y.index[id_split - self.step_ahead + i + 2]] = bernoulli.rvs((lambda p: p if 1>=p>=0 else ( 1 if p>1 else 0 ))(p_hat), size=1)[0]
                    X_train_US = self.X.iloc[0:id_split, :]

                    X_test_US = self.X.iloc[id_split:, :]

                    model = LogisticRegression(max_iter=1000).fit(X_train_US, Y_train_US_hat)
                    self.Y_test_probs.iloc[id_split, -1] = model.predict_proba(X_test_US)[0][1]
                    self.Y_test_label.iloc[id_split, -1] = model.predict(X_test_US)[0]

            elif threshold_tuning:
                print("predicting  Y_%i | X[0:%i,:], threshold tuning" % (id_split, id_split - self.step_ahead + 1))
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
                X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

                X_test_US = self.X.iloc[id_split:, :]

                model = LogisticRegression(max_iter=1000).fit(X_train_US, Y_train_US)
                self.Y_test_probs.iloc[id_split, -1] = model.predict_proba(X_test_US)[0][1]
                self.Y_test_label.iloc[id_split, -1] = lambda: 1 if model.predict(X_test_US)[0] >= 0.5 else 0
            else:
                print("predicting  Y_%i | X[0:%i,:] expanding window pca" % (id_split, id_split - self.step_ahead + 1))
                data=self.X.iloc[0:id_split - self.step_ahead + 1, :]
                most_important_features = self.rolling_PCA(data=data,important_features=True)
                data_tr,data_ts=data.filter(most_important_features),self.X.iloc[id_split:, :].filter(most_important_features)
                # X_train_US = data_tr
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
                # X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]
                # X_test_US = self.X.iloc[id_split:, :]

                model = LogisticRegression(max_iter=1000).fit(data_tr, Y_train_US)
                self.Y_test_probs.iloc[id_split, -1] = model.predict_proba(data_ts)[0][1]
                self.Y_test_label.iloc[id_split, -1] = model.predict(data_ts)[0]


    def RF_model(self,meth_1=True):

        print(str(self.step_ahead) + " step-ahead training and predicting with RF model..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:
            if meth_1:
                print("predicting  Y_%i | X[0:%i,:] + method_1" % (id_split, id_split - self.step_ahead + 1))
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
                Y_hat_US_labs, Y_hat_US_probs = Y_train_US.copy(), Y_train_US.copy()
                if id_split < 431:
                    for i in range(self.step_ahead):
                        X_train_US = self.X.iloc[0:id_split - self.step_ahead + i + 1, :]
                        X_test_US = self.X.iloc[id_split - self.step_ahead + i + 2:, :]
                        Y_train_tilde = Y_hat_US_labs.iloc[0:id_split - self.step_ahead + i + 1]

                        model = RandomForestClassifier(n_estimators=2000, random_state=42).fit(X_train_US,
                                                                                               Y_train_tilde)
                        Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = \
                            model.predict_proba(X_test_US)[0][1]
                        Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = model.predict(X_test_US)[0]

                if id_split < 431:
                    self.Y_test_probs.iloc[id_split, 1] = Y_hat_US_probs[id_split]
                    self.Y_test_label.iloc[id_split, 1] = Y_hat_US_labs[id_split]
            else:
                print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
                X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

                X_test_US = self.X.iloc[id_split:, :]

                model = RandomForestClassifier(n_estimators=2000, random_state=42).fit(X_train_US, Y_train_US)
                self.var_imp.iloc[id_split, :] = model.feature_importances_ * 100
                self.Y_test_probs.iloc[id_split,1] = model.predict_proba(X_test_US)[0][1]
                self.Y_test_label.iloc[id_split, 1] = model.predict(X_test_US)[0]

    def EN_model(self, f=True):
        print(str(self.step_ahead) + " step-ahead training and predicting with Elastic Net model..")
        range_data_split = range(self.date_split, len(self.X))
        if f:
            for id_split in range_data_split:

                print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
                X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

                X_test_US = self.X.iloc[id_split:, :]

                model = ElasticNet().fit(X_train_US, Y_train_US)
                # self.var_imp.iloc[id_split, :] = en.feature_importances_ * 100
                self.Y_test_probs.iloc[id_split,1] = model.predict(X_test_US)[0]
                self.Y_test_label.iloc[id_split, 1] = int(model.predict(X_test_US)[0] >= 0.5)
        else:
            X_train_US,X_test_US,Y_train_US,Y_test_US = train_test_split(self.X,self.Y,train_size=204, shuffle= False)
            model = LogisticRegression(penalty="elasticnet", l1_ratio=1, max_iter=1000, solver="saga").fit(X_train_US, Y_train_US)
            # self.var_imp.iloc[id_split, :] = en.feature_importances_ * 100
            print(len(X_test_US))
            for id in range(len(X_test_US)):
                self.Y_test_label.iloc[(id+204), 1] = model.predict(X_test_US)[id]
                self.Y_test_probs.iloc[(id+204), 1] = model.predict_proba(X_test_US)[id][1]



    def ABC_model(self):
        print(str(self.step_ahead) + "step-ahead training and predicting with AdaBoostClassifier model..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:

            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
            X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

            X_test_US = self.X.iloc[id_split:, :]

            model = AdaBoostRegressor(n_estimators=500, learning_rate=0.1,base_estimator=LogisticRegression()).fit(X_train_US, Y_train_US)
            # self.var_imp.iloc[id_split, :] = model.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split,1] = model.predict(X_test_US)[0]
            self.Y_test_label.iloc[id_split, 1] = model.predict(X_test_US)[0]

    def GB_model(self):
        print(str(self.step_ahead) + "step-ahead training and predicting with GradientBoosting model..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:

            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
            X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

            X_test_US = self.X.iloc[id_split:, :]

            model = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.1, random_state=42).fit(X_train_US, Y_train_US)
            self.Y_test_probs.iloc[id_split,1] = model.predict(X_test_US)[0]
            self.Y_test_label.iloc[id_split, 1] = model.predict(X_test_US)[0]

    def BC_model(self):
        print(str(self.step_ahead) + "step-ahead training and predicting with BaggingClassifier model..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:

            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
            X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

            X_test_US = self.X.iloc[id_split:, :]

            model = BaggingClassifier(n_estimators=2000, random_state=42).fit(X_train_US, Y_train_US)
            # self.var_imp.iloc[id_split, :] = model.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split,1] = model.predict_proba(X_test_US)[0][1]
            self.Y_test_label.iloc[id_split, 1] = model.predict(X_test_US)[0]
    def KNN_model(self):
        print(str(self.step_ahead) + "step-ahead training and predicting with KNN model..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:
            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1))
            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
            X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

            X_test_US = self.X.iloc[id_split:, :]

            model = KNeighborsRegressor(n_neighbors=1).fit(X_train_US, Y_train_US)
            # self.var_imp.iloc[id_split, :] = model.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split, 1] = model.predict(X_test_US)[0]
            self.Y_test_label.iloc[id_split, 1] = int(model.predict(X_test_US)[0]>0.5)
    def DTR_model(self):
        print(str(self.step_ahead) + "step-ahead training and predicting with DTR model..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:

            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
            X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

            X_test_US = self.X.iloc[id_split:, :]

            model = DecisionTreeRegressor(random_state=42,criterion="friedman_mse").fit(X_train_US, Y_train_US)
            # self.var_imp.iloc[id_split, :] = model.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split,1] = model.predict(X_test_US)[0]
            self.Y_test_label.iloc[id_split, 1] = int(model.predict(X_test_US)[0]>=0.5)

    def CV_RF_model(self):
        print(str(self.step_ahead) + " step-ahead training and predicting with Cross Validation RF..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:
            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1))
            params_grid_forest = {'max_depth': [None] + list(range(4, 8, 2)),  # The maximum depth of the tree.
                                  'min_samples_split': range(2, 8, 2),
                                  # [2, 4, 6] #The minimum number of samples required to split an internal node
                                  'n_estimators': [100, 200]}  # Number of classifiers (decision trees here)

            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
            X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

            X_test_US = self.X.iloc[id_split:, :]
            grid_search_cv_forest = GridSearchCV(RandomForestClassifier(random_state=42), params_grid_forest,
                                                 scoring="accuracy", cv=5)
            model_forest = grid_search_cv_forest.fit(X_train_US, Y_train_US)
            print(model_forest.best_params_)
            self.var_imp.iloc[id_split, :] = model_forest.best_estimator_.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split, 1] = model_forest.predict_proba(X_test_US)[0][1]
            self.Y_test_label.iloc[id_split, 1] = model_forest.predict(X_test_US)[0]

#if __name__ == '__main__':
#   model_instance = Models()
#    model_instance.split(0.2)
#    model_instance.fit()
#    print(model_instance.predict())
#    print("Accuracy: ",     model_instance.model.score(model_instance.X_test, model_instance.y_test))
