import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from statistics import mean
from collections import Counter
from scipy.stats import bernoulli
from sklearn.utils import shuffle


# CLassification
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

#Neural Network Classifier
from sklearn.neural_network import MLPClassifier

#Performance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,\
    roc_auc_score, roc_curve, confusion_matrix ,mean_squared_error, precision_recall_curve, log_loss

# Cross validation
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

#Data processing
from Main.Data_processing import standardization_norm, PCA_, minmax_norm, resample_dataframe

class Models:

    def __init__(self,name:str, Y, X, date_split: int, step_ahead: int, method= None):
        """
        Initialisation de l'objet "models" et des variables qui seront utiles à la prédiction

        :param name: Nom du modèle (cf. la fonction predict())
        :param Y: La variable à prédire
        :param X: La matrice des covariables
        :param date_split: La date à partir de laquelle on prédit les Y
        :param step_ahead: Le pas
        """

        self.name = name
        self.method = method

        if self.method == "method_1":
            self.name += "_and_method_1"
        elif self.method == "method_2":
            self.name += "_and_method_2"
        elif self.method == "threshold_tuning":
            self.name += "_and_threshold_tuning"
        elif self.method == "pca":
            self.name += "_and_expanding_PCA"
        self.date_split = date_split
        self.step_ahead = step_ahead
        self.X = X
        self.Y = Y
        self.Y_hat = pd.concat(
            [self.Y,pd.DataFrame(np.nan, index=self.Y.index, columns=['label']),
             pd.DataFrame(np.nan, index=self.Y.index, columns=["probs"])],
            axis=1)
        self.var_imp = pd.DataFrame(np.nan, index=X.index, columns=self.X.columns)




    def predict(self):
        if self.name.split('_')[0] == "logit":
            return self.logit_model()
        elif self.name.split('_')[0] == "RF":
            return self.RF_model()
        elif self.name.split('_')[0] == "EN":
            return self.EN_model()
        elif self.name.split('_')[0] == "ABC":
            return self.ABC_model()
        elif self.name.split('_')[0] == "CV_RF":
            return self.CV_RF_model()
        elif self.name.split('_')[0] == "GB":
            return self.GB_model()
        elif self.name.split('_')[0] == "BC":
            return self.BC_model()
        elif self.name.split('_')[0] == "DTR":
            return self.DTR_model()
        elif self.name.split('_')[0] =="KNN":
            return self.KNN_model()
        elif self.name.split('_')[0] =="MLP":
            return self.MLP_model()
        aggregate_var_imp_RF_v1 = pd.DataFrame(np.nansum(self.var_imp, axis=0).T, index=self.X.columns, columns=['Importance'])
        aggregate_var_imp_RF_v1.index.name = 'variables'
        print(aggregate_var_imp_RF_v1.sort_values(by="Importance", ascending=False).head(10))


    def logit_model(self):
        method_1, method_2, threshold_tuning, pca = True, False ,False ,False
        if self.method == "method_1":
            method_1 = True
        elif self.method == "method_2":
            method_2 = True
        elif self.method == "threshold_tuning":
            threshold_tuning = True
        elif self.method == "pca":
            pca= True

        print(str(self.step_ahead) + " step-ahead training and predicting with " + self.name)



        range_data_split = range(self.date_split, len(self.X))
        most_imp=[]
        naif = False
        for id_split in range_data_split:

            # This method consists of predicting missed Y's that we've
            # supposed unknown for 12 months before our current prediction

            if method_1:

                normalize, resample, threshold_tuning, pca, params_tuning = True, False, False, False, False

                print("predicting  Y_%i | X[0:%i,:] with method 1" % (id_split, id_split - self.step_ahead + 1 ))
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]

                # Declare variables used to store predictions of missed values of Y

                Y_hat_US_labs, Y_hat_US_probs = Y_train_US.copy(), Y_train_US.copy()


                if id_split<431:

                    # Loop on missed values of Y

                    for i in range(self.step_ahead):
                        # Train set goes to id of our initial train set to our last predicted/known missing Y

                        print("\t prediciting Y_t- %i" % (12-i))

                        Y_train_tilde = Y_hat_US_labs.iloc[0:id_split - self.step_ahead + i + 1]
                        X_train_US = self.X.iloc[0:id_split - self.step_ahead + i + 1, :]
                        X_test_US = self.X.iloc[id_split - self.step_ahead + i + 2:, :]

                        # If True resamples train data
                        # Check function implementation for more details

                        if resample:
                            resampled = resample_dataframe(pd.concat([X_train_US,Y_train_tilde], axis=1))
                            Y_train_tilde = resampled.iloc[:, -1]

                            X_train_US = resampled.iloc[:, :-1]


                        # If True normalizes data

                        if normalize:

                            # Here we can also use standardization_norm()

                            X_train_US, X_test_US = minmax_norm(X_train_US,X_test_US)

                            # Drop nan observations generated by normalization

                            if X_train_US.isna().any().any():
                                aux = pd.concat([X_train_US,Y_train_tilde],axis=1).dropna()
                                Y_train_tilde = aux.iloc[:,-1]
                                X_train_US = aux.iloc[:,:-1]



                        # If True performs a PCA on train set

                        if pca:

                            most_important_features = PCA_(data=X_train_US, important_features=False, n_comp=.99)
                            most_imp.append(most_important_features)
                            X_train_US, X_test_US = X_train_US.filter(most_important_features), X_test_US.filter(most_important_features)

                        # If True selects the best hyperparams of the model
                        if params_tuning:
                            params_grid = {'C': [0.001,0.01,0.1,1,10,100,1000]}
                            grid_search_cv= GridSearchCV(LogisticRegression(max_iter=5000,random_state=42),
                                                                params_grid,
                                                                scoring="neg_log_loss", cv=5)
                            model = grid_search_cv.fit(X_train_US, Y_train_tilde)
                            print(model.best_params_)

                        # Create model for each iteration to predict the next unknown Y

                        else:
                            model = LogisticRegression(max_iter=5000,random_state=42, C = 10).fit(X_train_US,Y_train_tilde)

                        if threshold_tuning:

                            if id_split - 204 > 100:

                                print("threshold tuning for the %i th observation" % (id_split+i))
                                # Define thresholds

                                thresholds = np.arange(0.3, 1, 0.01)

                                # Define previous true Y's and its associated X

                                previous_true_Y = self.Y.iloc[self.date_split + 12:id_split - 12]

                                associated_X = self.X.iloc[self.date_split:id_split - 24:, :]

                                if normalize:

                                    associated_X = minmax_norm(associated_X,associated_X)[0]

                                if pca:
                                    associated_X = associated_X.filter(most_important_features)
                                # Evaluate each threshold

                                scores = [log_loss(previous_true_Y, (
                                        model.predict_proba(associated_X)[:,
                                        1] >= t).astype('int')) for t in thresholds]

                                # Get the best threshold according to the score

                                ix = np.argmin(scores)
                                opt_threshold = thresholds[ix]

                                Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = model.predict_proba(X_test_US)[0][1]
                                Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = (lambda t: 1 if model.predict_proba(X_test_US)[0][1] >= t else 0)(opt_threshold)

                                print('Threshold=%.3f, Logloss-Score=%.5f' % (opt_threshold, scores[ix]))
                            else:

                                # Store prediction of missed Ys

                                Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = \
                                model.predict_proba(X_test_US)[0][1]
                                Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = \
                                model.predict(X_test_US)[0]

                        else:

                            # Store prediction of missed Ys

                            Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = model.predict_proba(X_test_US)[0][1]
                            Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = model.predict(X_test_US)[0]

                if id_split < 431:
                    self.Y_hat["probs"].iloc[id_split] = Y_hat_US_probs[id_split]
                    self.Y_hat["label"].iloc[id_split] = Y_hat_US_labs[id_split]
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
                    self.Y_hat["probs"].iloc[id_split] = model.predict_proba(X_test_US)[0][1]
                    self.Y_hat["label"].iloc[id_split] = model.predict(X_test_US)[0]
                else:
                    print(Counter(Y_train_US[id_split - self.step_ahead-24:]).most_common())
                    if Counter(Y_train_US[id_split - self.step_ahead-24:]).most_common()[0][0]==0 and Counter(Y_train_US[id_split - self.step_ahead-24:]).most_common()[0][1]>=17:
                        print(Counter(Y_train_US[id_split - self.step_ahead-24:]).most_common()[0])
                        p_hat = mean(Y_train_US[id_split - self.step_ahead-24:]) +.3
                    elif Counter(Y_train_US[id_split - self.step_ahead-3*12:]).most_common()[0][0]==1 and Counter(Y_train_US[id_split - self.step_ahead-3*12:]).most_common()[0][1]>=3*12:
                        print(Counter(Y_train_US[id_split - self.step_ahead-24:]).most_common()[0])
                        p_hat = mean(Y_train_US[id_split - self.step_ahead-24:]) -.3
                    else:
                        p_hat = mean(Y_train_US[id_split - self.step_ahead-24:])
                    print(p_hat,mean(Y_train_US[id_split - self.step_ahead-24:]))
                    for i in range(self.step_ahead-1):
                        Y_train_US_hat[self.Y.index[id_split - self.step_ahead + i + 2]] = bernoulli.rvs((lambda p: p if 1>=p>=0 else ( 1 if p>1 else 0 ))(p_hat), size=1)[0]
                    X_train_US = self.X.iloc[0:id_split, :]

                    X_test_US = self.X.iloc[id_split:, :]

                    model = LogisticRegression(max_iter=1000).fit(X_train_US, Y_train_US_hat)
                    self.Y_hat["probs"].iloc[id_split] = model.predict_proba(X_test_US)[0][1]
                    self.Y_hat["label"].iloc[id_split] = model.predict(X_test_US)[0]

            elif threshold_tuning:
                print("predicting  Y_%i | X[0:%i,:], threshold tuning" % (id_split, id_split - self.step_ahead + 1))
                df = resample_dataframe(pd.concat(
                    [self.X.iloc[0:id_split - self.step_ahead + 1, :], self.Y.iloc[0:id_split - self.step_ahead + 1]],
                    axis=1))
                Y_train_US = df.iloc[:, -1]
                X_train_US = df.iloc[:, :-1]

                X_test_US = self.X.iloc[id_split:, :]

                model = LogisticRegression(max_iter=5000).fit(X_train_US, Y_train_US)

                self.Y_hat["probs"].iloc[id_split] = model.predict_proba(X_test_US)[0][1]
                self.Y_hat["label"].iloc[id_split] = (lambda t: 1 if model.predict_proba(X_test_US)[0][1] >= t else 0)(0.5)
                b=True
                if b==True:
                    if id_split-204>100:
                        # define thresholds
                        thresholds = np.arange(0.3, 1, 0.01)
                        # evaluate each threshold
                        scores = [log_loss(self.Y.iloc[self.date_split+12:id_split-12], (model.predict_proba(self.X.iloc[self.date_split:id_split-24:, :])[:,1]>=t).astype('int')) for t in thresholds]
                        # get best threshold
                        ix = np.argmin(scores)
                        opt_threshold=thresholds[ix]

                        self.Y_hat["probs"].iloc[id_split]= model.predict_proba(X_test_US)[0][1]
                        self.Y_hat["label"].iloc[id_split] = (lambda t: 1 if model.predict_proba(X_test_US)[0][1] >= t else 0)(opt_threshold)
                        print('Threshold=%.3f, Logloss-Score=%.5f' % (opt_threshold, scores[ix]))
                else:
                    if id_split - 204 > 100:
                        precision, recall, thresholds = precision_recall_curve(self.Y.iloc[self.date_split+12:id_split-12], model.predict_proba(self.X.iloc[self.date_split:id_split-24:, :])[:,1])
                        # calculate the g-mean for each threshold
                        #gmeans = np.sqrt(tpr * (1 - fpr))
                        fscore = (2 * precision * recall) / (precision + recall)
                        # locate the index of the largest g-mean
                        ix = np.argmax(fscore)
                        print('Best Threshold=%f, fscore=%.3f' % (thresholds[ix], fscore[ix]))
                        # plot the roc curve for the model

                        # show the plot
                        #plt.show()
                        opt_threshold = thresholds[ix]
                        self.Y_hat["probs"].iloc[id_split] = model.predict_proba(X_test_US)[0][1]
                        self.Y_hat["label"].iloc[id_split] = (
                            lambda t: 1 if model.predict_proba(X_test_US)[0][1] >= t else 0)(opt_threshold)
            elif pca:
                print("predicting  Y_%i | X[0:%i,:] expanding window pca" % (id_split, id_split - self.step_ahead + 1))
                data=self.X.iloc[0:id_split - self.step_ahead + 1, :]
                most_important_features = PCA_(data=data,important_features=False,n_comp=22)
                most_imp.append(most_important_features)
                data_tr,data_ts=data.filter(most_important_features),self.X.iloc[id_split:, :].filter(most_important_features)
                # X_train_US = data_tr
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
                # X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]
                # X_test_US = self.X.iloc[id_split:, :]

                model = LogisticRegression(max_iter=5000).fit(data_tr, Y_train_US)
                self.Y_hat["probs"].iloc[id_split] = model.predict_proba(data_ts)[0][1]
                self.Y_hat["label"].iloc[id_split] = model.predict(data_ts)[0]

            else:

                print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1))
                df = resample_dataframe(pd.concat([self.X.iloc[0:id_split - self.step_ahead + 1, :],self.Y.iloc[0:id_split - self.step_ahead + 1]],axis=1))
                Y_train_US = df.iloc[:,-1]
                X_train_US = df.iloc[:,:-1]
                X_test_US = self.X.iloc[id_split:, :]

                X_train_US_normalized, X_test_US_normalized = standardization_norm(X_train_US,X_test_US)


                model = LogisticRegression(max_iter=5000,random_state=42).fit(X_train_US_normalized, Y_train_US)

                self.Y_hat["probs"].iloc[id_split]= model.predict_proba(X_test_US_normalized)[0][1]
                self.Y_hat["label"].iloc[id_split] = model.predict(X_test_US_normalized)[0]

        if pca:
            return model,self.Y_hat, most_imp[-1]
        else:
            return model, self.Y_hat


    def RF_model(self,meth_1=False, threshold_tuning=True):

        print(str(self.step_ahead) + " step-ahead training and predicting with RF model..")
        range_data_split = range(self.date_split, len(self.X)- self.step_ahead)
        method_1=True
        most_imp=[]
        for id_split in range_data_split:
            if method_1:

                normalize, resample, threshold_tuning, pca, params_tuning = True, True, False, False, False

                print("predicting  Y_%i | X[0:%i,:] with method 1" % (id_split, id_split - self.step_ahead + 1))
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]

                # Declare variables used to store predictions of missed values of Y

                Y_hat_US_labs, Y_hat_US_probs = Y_train_US.copy(), Y_train_US.copy()

                if id_split < 431:

                    # Loop on missed values of Y

                    for i in range(self.step_ahead):
                        # Train set goes to id of our initial train set to our last predicted/known missing Y

                        print("\t prediciting Y_t- %i" % (12 - i))

                        Y_train_tilde = Y_hat_US_labs.iloc[0:id_split - self.step_ahead + i + 1]
                        X_train_US = self.X.iloc[0:id_split - self.step_ahead + i + 1, :]
                        X_test_US = self.X.iloc[id_split - self.step_ahead + i + 2:, :]

                        # If True resamples train data
                        # Check function implementation for more details

                        if resample:
                            resampled = resample_dataframe(pd.concat([X_train_US, Y_train_tilde], axis=1))
                            Y_train_tilde = resampled.iloc[:, -1]

                            X_train_US = resampled.iloc[:, :-1]

                        # If True normalizes data

                        if normalize:

                            # Here we can also use standardization_norm()

                            X_train_US, X_test_US = minmax_norm(X_train_US, X_test_US)

                            # Drop nan observations generated by normalization

                            if X_train_US.isna().any().any():
                                aux = pd.concat([X_train_US, Y_train_tilde], axis=1).dropna()
                                Y_train_tilde = aux.iloc[:, -1]
                                X_train_US = aux.iloc[:, :-1]

                        # If True performs a PCA on train set

                        if pca:
                            most_important_features = PCA_(data=X_train_US, important_features=False, n_comp=.99)
                            most_imp.append(most_important_features)
                            X_train_US, X_test_US = X_train_US.filter(most_important_features), X_test_US.filter(
                                most_important_features)

                        # If True selects the best hyperparams of the model
                        if params_tuning:
                            params_grid = {'max_depth': [None] + list(range(4, 8, 2)),
                                  'min_samples_split': range(2, 8, 2),
                                  'n_estimators': [100, 200]}
                            grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=42),
                                                          params_grid,
                                                          scoring="neg_log_loss", cv=5)
                            model = grid_search_cv.fit(X_train_US, Y_train_tilde)
                            print(model.best_params_)

                        # Create model for each iteration to predict the next unknown Y

                        else:
                            model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train_US,
                                                                                                 Y_train_tilde)

                        if threshold_tuning:

                            if id_split - 204 > 100:

                                print("threshold tuning for the %i th observation" % (id_split + i))
                                # Define thresholds

                                thresholds = np.arange(0.3, 1, 0.01)

                                # Define previous true Y's and its associated X

                                previous_true_Y = self.Y.iloc[self.date_split + 12:id_split - 12]

                                associated_X = self.X.iloc[self.date_split:id_split - 24:, :]

                                if normalize:
                                    associated_X = minmax_norm(associated_X, associated_X)[0]

                                if pca:
                                    associated_X = associated_X.filter(most_important_features)
                                # Evaluate each threshold

                                scores = [log_loss(previous_true_Y, (
                                        model.predict_proba(associated_X)[:,
                                        1] >= t).astype('int')) for t in thresholds]

                                # Get the best threshold according to the score

                                ix = np.argmin(scores)
                                opt_threshold = thresholds[ix]

                                Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = \
                                model.predict_proba(X_test_US)[0][1]
                                Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = (
                                    lambda t: 1 if model.predict_proba(X_test_US)[0][1] >= t else 0)(opt_threshold)

                                print('Threshold=%.3f, Logloss-Score=%.5f' % (opt_threshold, scores[ix]))
                            else:

                                # Store prediction of missed Ys

                                Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = \
                                    model.predict_proba(X_test_US)[0][1]
                                Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = \
                                    model.predict(X_test_US)[0]

                        else:

                            # Store prediction of missed Ys

                            Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = \
                            model.predict_proba(X_test_US)[0][1]
                            Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = model.predict(X_test_US)[
                                0]

                if id_split < 431:
                    self.Y_hat["probs"].iloc[id_split] = Y_hat_US_probs[id_split]
                    self.Y_hat["label"].iloc[id_split] = Y_hat_US_labs[id_split]
            else:
                print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]
                X_train_US = self.X.iloc[0:id_split - self.step_ahead + 1, :]

                X_test_US = self.X.iloc[id_split:, :]

                model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train_US, Y_train_US)
                self.var_imp.iloc[id_split, :] = model.feature_importances_ * 100
                self.Y_hat["probs"].iloc[id_split] = model.predict_proba(X_test_US)[0][1]
                self.Y_hat["label"].iloc[id_split] = model.predict(X_test_US)[0]
        return model
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
                self.Y_hat["probs"].iloc[id_split]= model.predict(X_test_US)[0]
                self.Y_hat["label"].iloc[id_split]= int(model.predict(X_test_US)[0] >= 0.5)

    def ABC_model(self):
        print(str(self.step_ahead) + "step-ahead training and predicting with AdaBoostClassifier model..")
        range_data_split = range(self.date_split, len(self.X))
        method_1= True
        most_imp = []
        for id_split in range_data_split:

            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
            if method_1:

                normalize, resample, threshold_tuning, pca, params_tuning = True, False, False, False, False

                print("predicting  Y_%i | X[0:%i,:] with method 1" % (id_split, id_split - self.step_ahead + 1))
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]

                # Declare variables used to store predictions of missed values of Y

                Y_hat_US_labs, Y_hat_US_probs = Y_train_US.copy(), Y_train_US.copy()

                if id_split < 431:

                    # Loop on missed values of Y

                    for i in range(self.step_ahead):
                        # Train set goes to id of our initial train set to our last predicted/known missing Y

                        print("\t prediciting Y_t- %i" % (12 - i))

                        Y_train_tilde = Y_hat_US_labs.iloc[0:id_split - self.step_ahead + i + 1]
                        X_train_US = self.X.iloc[0:id_split - self.step_ahead + i + 1, :]
                        X_test_US = self.X.iloc[id_split - self.step_ahead + i + 2:, :]

                        # If True resamples train data
                        # Check function implementation for more details

                        if resample:
                            resampled = resample_dataframe(pd.concat([X_train_US, Y_train_tilde], axis=1))
                            Y_train_tilde = resampled.iloc[:, -1]

                            X_train_US = resampled.iloc[:, :-1]

                        # If True normalizes data

                        if normalize:

                            # Here we can also use standardization_norm()

                            X_train_US, X_test_US = minmax_norm(X_train_US, X_test_US)

                            # Drop nan observations generated by normalization

                            if X_train_US.isna().any().any():
                                aux = pd.concat([X_train_US, Y_train_tilde], axis=1).dropna()
                                Y_train_tilde = aux.iloc[:, -1]
                                X_train_US = aux.iloc[:, :-1]

                        # If True performs a PCA on train set

                        if pca:
                            most_important_features = PCA_(data=X_train_US, important_features=False, n_comp=.99)
                            most_imp.append(most_important_features)
                            X_train_US, X_test_US = X_train_US.filter(most_important_features), X_test_US.filter(
                                most_important_features)

                        # If True selects the best hyperparams of the model
                        if params_tuning:
                            params_grid = {'n_estimators': [100,200,300,500],
                                           'learning_rate': np.arange(0,1,0.01)}
                            grid_search_cv = GridSearchCV(AdaBoostRegressor(n_estimators=500, learning_rate=0.1,base_estimator=LogisticRegression()),
                                                          params_grid,
                                                          scoring="neg_log_loss", cv=5)
                            model = grid_search_cv.fit(X_train_US, Y_train_tilde)
                            print(model.best_params_)

                        # Create model for each iteration to predict the next unknown Y

                        else:
                            model = AdaBoostRegressor(n_estimators=500, learning_rate=0.1,base_estimator=LogisticRegression()).fit(X_train_US,
                                                                                                 Y_train_tilde)

                        if threshold_tuning:

                            if id_split - 204 > 100:

                                print("threshold tuning for the %i th observation" % (id_split + i))
                                # Define thresholds

                                thresholds = np.arange(0.3, 1, 0.01)

                                # Define previous true Y's and its associated X

                                previous_true_Y = self.Y.iloc[self.date_split + 12:id_split - 12]

                                associated_X = self.X.iloc[self.date_split:id_split - 24:, :]

                                if normalize:
                                    associated_X = minmax_norm(associated_X, associated_X)[0]

                                if pca:
                                    associated_X = associated_X.filter(most_important_features)
                                # Evaluate each threshold

                                scores = [log_loss(previous_true_Y, (
                                        model.predict_proba(associated_X)[:,
                                        1] >= t).astype('int')) for t in thresholds]

                                # Get the best threshold according to the score

                                ix = np.argmin(scores)
                                opt_threshold = thresholds[ix]

                                Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = \
                                model.predict_proba(X_test_US)[0][1]
                                Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = (
                                    lambda t: 1 if model.predict_proba(X_test_US)[0][1] >= t else 0)(opt_threshold)

                                print('Threshold=%.3f, Logloss-Score=%.5f' % (opt_threshold, scores[ix]))
                            else:

                                # Store prediction of missed Ys

                                Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = \
                                    model.predict_proba(X_test_US)[0][1]
                                Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = \
                                    model.predict(X_test_US)[0]

                        else:

                            # Store prediction of missed Ys

                            Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = \
                            model.predict_proba(X_test_US)[0][1]
                            Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = model.predict(X_test_US)[
                                0]

                if id_split < 431:
                    self.Y_hat["probs"].iloc[id_split] = Y_hat_US_probs[id_split]
                    self.Y_hat["label"].iloc[id_split] = Y_hat_US_labs[id_split]



    def GB_model(self):
        print(str(self.step_ahead) + "step-ahead training and predicting with GradientBoosting model..")
        range_data_split = range(self.date_split, len(self.X))
        method_1=True
        most_imp=[]
        for id_split in range_data_split:

            if method_1:

                normalize, resample, pca, params_tuning = True, True, False, False

                print("predicting  Y_%i | X[0:%i,:] with method 1" % (id_split, id_split - self.step_ahead + 1))
                Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]

                # Declare variables used to store predictions of missed values of Y

                Y_hat_US_labs, Y_hat_US_probs = Y_train_US.copy(), Y_train_US.copy()

                if id_split < 431:

                    # Loop on missed values of Y

                    for i in range(self.step_ahead):
                        # Train set goes to id of our initial train set to our last predicted/known missing Y

                        print("\t prediciting Y_t- %i" % (12 - i))

                        Y_train_tilde = Y_hat_US_labs.iloc[0:id_split - self.step_ahead + i + 1]
                        X_train_US = self.X.iloc[0:id_split - self.step_ahead + i + 1, :]
                        X_test_US = self.X.iloc[id_split - self.step_ahead + i + 2:, :]

                        # If True resamples train data
                        # Check function implementation for more details

                        if resample:
                            resampled = resample_dataframe(pd.concat([X_train_US, Y_train_tilde], axis=1))
                            Y_train_tilde = resampled.iloc[:, -1]

                            X_train_US = resampled.iloc[:, :-1]

                        # If True normalizes data

                        if normalize:

                            # Here we can also use standardization_norm()

                            X_train_US, X_test_US = minmax_norm(X_train_US, X_test_US)

                            # Drop nan observations generated by normalization

                            if X_train_US.isna().any().any():
                                aux = pd.concat([X_train_US, Y_train_tilde], axis=1).dropna()
                                Y_train_tilde = aux.iloc[:, -1]
                                X_train_US = aux.iloc[:, :-1]

                        # If True performs a PCA on train set

                        if pca:
                            most_important_features = PCA_(data=X_train_US, important_features=False, n_comp=.99)
                            most_imp.append(most_important_features)
                            X_train_US, X_test_US = X_train_US.filter(most_important_features), X_test_US.filter(
                                most_important_features)

                        # If True selects the best hyperparams of the model
                        if params_tuning:
                            params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                            grid_search_cv = GridSearchCV(LogisticRegression(max_iter=5000, random_state=42),
                                                          params_grid,
                                                          scoring="neg_log_loss", cv=5)
                            model = grid_search_cv.fit(X_train_US, Y_train_tilde)
                            print(model.best_params_)

                        # Create model for each iteration to predict the next unknown Y

                        else:
                            model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42).fit(X_train_US,
                                                                                                 Y_train_tilde)




                        # Store prediction of missed Ys

                        Y_hat_US_probs[self.Y.index[id_split - self.step_ahead + i + 2]] = model.predict(X_test_US)[0]
                        Y_hat_US_labs[self.Y.index[id_split - self.step_ahead + i + 2]] = model.predict(X_test_US)[
                            0]

                if id_split < 431:
                    self.Y_hat["probs"].iloc[id_split] = Y_hat_US_probs[id_split]
                    self.Y_hat["label"].iloc[id_split] = Y_hat_US_labs[id_split]
        return model, self.Y_hat

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


#Partie LSTM/MLP:
    def MLP_model(self):
        print(str(self.step_ahead) + "step-ahead training and predicting with MLPClassifier model..")
        range_data_split = range(self.date_split, len(self.X))
        for id_split in range_data_split:
            print("predicting  Y_%i | X[0:%i,:]" % (id_split, id_split - self.step_ahead + 1 ))
            data = self.X.iloc[0:id_split - self.step_ahead + 1, :]
            most_important_features = self.PCA_(data=data, important_features=True)
            data_tr, data_ts = data.filter(most_important_features), self.X.iloc[id_split:, :].filter(
                most_important_features)
            Y_train_US = self.Y.iloc[0:id_split - self.step_ahead + 1]

            model = MLPClassifier(random_state=42, max_iter=1000, activation="logistic").fit(data_tr, Y_train_US)
            # self.var_imp.iloc[id_split, :] = model.feature_importances_ * 100
            self.Y_test_probs.iloc[id_split,1] = model.predict_proba(data_ts)[0][1]
            self.Y_test_label.iloc[id_split, 1] = model.predict(data_ts)[0]
