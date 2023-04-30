import pandas as pd
import numpy as np
import warnings



# CLassification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsRegressor

#Performance
from sklearn.metrics import accuracy_score

# Cross validation
from sklearn.model_selection import GridSearchCV

from Main.Data_processing import standardization_norm, PCA_, minmax_norm, resample_dataframe
from utils.helpers import get_optimal_thresholds

class Models:
    """
    A class for initializing prediction models and their parameters.
    """

    def __init__(self, Y: pd.DataFrame, X: pd.DataFrame,
                 date_split: int, step_ahead: int, tuning: dict, method:dict) -> None:
        """
        Initialize the "models" object and the variables necessary for prediction.

        :param Y: The variable to be predicted
        :param X: The covariate matrix
        :param date_split: The date from which to predict Y
        :param step_ahead: The prediction step
        :param tuning: dictionary (boolean) of model tuning operations
        """

        self.date_split = date_split
        self.step_ahead = step_ahead
        self.tuning = tuning
        self.method = method
        self.X = X
        self.Y = pd.DataFrame(np.nan, index=Y.index, columns=["true_label"])
        self.Y["true_label"] = Y

        self.Y_hat = pd.DataFrame(np.nan, index=self.Y.index, columns=["true_label"])
        self.Y_hat["true_label"] = self.Y["true_label"]
        # Add new models here
        for model in ["logit_", "rf_", "gb_"]:
            self.Y_hat[model + 'label'] = self.Y.iloc[:date_split]
            self.Y_hat[model + 'probs'] = None
            self.Y_hat[model + "opt_threshold_log_score"] = None
            self.Y_hat[model + "opt_threshold_g_means"] = None
            self.Y_hat[model + "opt_threshold_J"] = None
            self.Y_hat[model + "opt_threshold_f1_score"] = None
            self.Y_hat[model + "opt_label_log_score"] = None
            self.Y_hat[model + "opt_label_g_means"] = None
            self.Y_hat[model + "opt_label_J"] = None
            self.Y_hat[model + "opt_label_f1_score"] = None

        # Stores most important variables

        self.var_imp = pd.DataFrame(np.nan, index=X.index, columns=self.X.columns)

        # Stores accuracy of imputation

        self.acc_f_time = pd.DataFrame(np.nan,index=Y.index, columns = ["logit_acc","rf_acc","gb_acc"])

    def models(self):

        # Booleans to choose which method of imputation we are using
        method_1, method_2 = self.method["method_1"], self.method["method_2"]

        # Booleans to choose methods to enhance/tune the model
        normalize, resample, threshold_tuning, pca, params_tuning =self.tuning["normalize"],self.tuning["resample"],\
                                                                   self.tuning["threshold_tuning"], self.tuning["pca"],\
                                                                   self.tuning["params_tuning"]




        print(str(self.step_ahead) + " step-ahead training and predicting...")

        range_data_split = range(self.date_split, len(self.X))
        most_imp = []

        for id_split in range_data_split:

            # This method consists of predicting missing Y's that we've
            # supposed unknown for 12 months before our current prediction

            if method_1:

                print("predicting  Y_%i | X[0:%i,:] with method 1, date:%s" % (id_split, id_split - self.step_ahead, self.Y.index[id_split].strftime('%Y-%m-%d')))
                # Every 12 months, we erase potential errors
                # we could have done due to imputation

                Y_train_tilde_logit = self.Y_hat.loc[self.Y_hat.index[0:id_split - self.step_ahead], "true_label"]
                Y_train_tilde_rf = self.Y_hat.loc[self.Y_hat.index[0:id_split - self.step_ahead], "true_label"]
                Y_train_tilde_gb = self.Y_hat.loc[self.Y_hat.index[0:id_split - self.step_ahead], "true_label"]

                # Loop on missed values of Y

                for i in range(0, self.step_ahead + 1):

                    if i != self.step_ahead:

                        print("\t predicting missing value Y_%i, date :%s" % (id_split - self.step_ahead + i, self.Y.index[id_split - self.step_ahead + i].strftime('%Y-%m-%d')))

                    else:

                        print("\t predicting Y_%i with imputation, date:%s" % (id_split - self.step_ahead + i, self.Y.index[id_split - self.step_ahead + i].strftime('%Y-%m-%d')))


                    X_train_US = self.X.iloc[0:id_split - self.step_ahead + i, :]
                    X_test_US = self.X.iloc[id_split - self.step_ahead + i:, :]


                    # If True normalizes data

                    if normalize:

                        # Here we can also use minmax_norm()

                        X_train_US, X_test_US = standardization_norm(X_train_US, X_test_US)[0].copy(), standardization_norm(X_train_US, X_test_US)[1].copy()

                        # Drop nan observations generated by normalization

                        if X_train_US.isna().any().any():

                            aux_logit = pd.concat([X_train_US, Y_train_tilde_logit], axis=1).dropna().copy()
                            Y_train_tilde_logit = aux_logit.iloc[:, -1].copy()
                            X_train_US = aux_logit.iloc[:, :-1].copy()

                            aux_rf = pd.concat([X_train_US, Y_train_tilde_rf], axis=1).dropna().copy()
                            Y_train_tilde_rf = aux_rf.iloc[:, -1].copy()
                            X_train_US = aux_rf.iloc[:, :-1].copy()

                            aux_gb = pd.concat([X_train_US, Y_train_tilde_gb], axis=1).dropna().copy()
                            Y_train_tilde_gb = aux_gb.iloc[:, -1].copy()
                            X_train_US = aux_gb.iloc[:, :-1].copy()







                    # If True resamples train data
                    # Check function implementation for more details

                    if resample:

                        # Auxiliary variables to keep the format of the Y_train_tile_model df
                        # Otherwise we get errors if resample is True

                        aux_logit = Y_train_tilde_logit.copy()
                        aux_rf = Y_train_tilde_rf.copy()
                        aux_gb = Y_train_tilde_gb.copy()

                        resampled_logit = resample_dataframe(pd.concat([X_train_US, Y_train_tilde_logit], axis=1))
                        resampled_rf = resample_dataframe(pd.concat([X_train_US, Y_train_tilde_rf], axis=1))
                        resampled_gb = resample_dataframe(pd.concat([X_train_US, Y_train_tilde_gb], axis=1))

                        Y_train_tilde_logit = resampled_logit.iloc[:, -1]
                        Y_train_tilde_rf = resampled_rf.iloc[:, -1]
                        Y_train_tilde_gb = resampled_gb.iloc[:, -1]

                        X_train_US = resampled_logit.iloc[:, :-1]


                    # If True performs a PCA on train set

                    if pca:
                        most_important_features = PCA_(data=X_train_US, important_features=False, n_comp=.99)
                        most_imp.append(most_important_features)
                        X_train_US, X_test_US = X_train_US.filter(most_important_features), X_test_US.filter(
                            most_important_features)

                    # If True selects the best hyperparams of the model
                    if params_tuning and id_split == len(self.X) - 1:
                        params_grid_logit = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                        params_grid_gb = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


                        params_grid_rf = {'max_depth': [None] + list(range(4, 8, 2)),
                                       'min_samples_split': range(2, 8, 2),
                                       'n_estimators': [100, 200, 1000]}
                        grid_search_cv_logit = GridSearchCV(LogisticRegression(max_iter=5000, random_state=42),
                                                      params_grid_logit,
                                                      scoring="neg_log_loss", cv=5)
                        grid_search_cv_gb = GridSearchCV(XGBClassifier(random_state=42),
                                                      params_grid_gb,
                                                      scoring="neg_log_loss", cv=5)
                        grid_search_cv_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                                                      params_grid_rf,
                                                      scoring="neg_log_loss", cv=5)
                        model_logit = grid_search_cv_logit.fit(X_train_US, Y_train_tilde_logit)
                        model_gb = grid_search_cv_gb.fit(X_train_US, Y_train_tilde_rf)
                        model_rf = grid_search_cv_rf.fit(X_train_US, Y_train_tilde_gb)


                    # Create model for each iteration to predict the next unknown Y

                    else:

                        model_logit = LogisticRegression(max_iter=5000, random_state=42, C=10).fit(X_train_US,
                                                                                             Y_train_tilde_logit)
                        model_rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train_US,
                                                                                         Y_train_tilde_rf)
                        model_gb = XGBClassifier(n_estimators=200, learning_rate=0.1,
                                                      random_state=42).fit(X_train_US,Y_train_tilde_gb)





                    models = {"logit_": model_logit, "rf_": model_rf, "gb_": model_gb}

                    if threshold_tuning:

                        warnings.warn("Threshold tuning not implemented yet for meth_1")


                        print("\t \t threshold tuning for the %i th observation" % id_split)

                        #opt_threshold = get_optimal_thresholds(models, X_train_US, Y_train_tilde_logit)

                    # Get back the nonresampled Y_train_tilde to modify it

                    if resample:

                        Y_train_tilde_logit = aux_logit.copy()
                        Y_train_tilde_rf = aux_rf.copy()
                        Y_train_tilde_gb = aux_gb.copy()


                    # Store predictions of missing Y's

                    Y_train_tilde_logit.loc[self.Y_hat.index[id_split - self.step_ahead + i]] = model_logit.predict(X_test_US)[
                    0].copy()
                    Y_train_tilde_rf.loc[self.Y_hat.index[id_split - self.step_ahead + i]] = model_rf.predict(X_test_US)[
                    0].copy()
                    Y_train_tilde_gb.loc[self.Y_hat.index[id_split - self.step_ahead + i]] = model_gb.predict(X_test_US)[
                    0].copy()

                # Variable to compute the accuracy of predicting the missing Y's

                true_y = self.Y_hat.loc[self.Y_hat.index[id_split - self.step_ahead: id_split], "true_label"]


                # At the end of loop, one can check that id_split == id_split - self.step_ahead + i

                for (model_name, model) in zip(models, models.values()):

                    self.Y_hat.loc[self.Y_hat.index[id_split - self.step_ahead + i], model_name + "probs"] = \
                        '{:.2f}'.format(model.predict_proba(X_test_US)[0][1].copy())
                    self.Y_hat.loc[self.Y_hat.index[id_split - self.step_ahead + i], model_name + "label"] = \
                    model.predict(X_test_US)[
                        0].copy()

                    # Compute accuracy of previous imputed Ys

                    imputed_y = self.Y_hat.loc[self.Y_hat.index[id_split - self.step_ahead: id_split], model_name + "label"]
                    acc = accuracy_score(true_y, imputed_y)
                    self.acc_f_time.loc[id_split, model_name+ "acc"] = acc

                    if threshold_tuning:
                        for model in ["logit_", "rf_", "gb_"]:
                            self.Y_hat.loc[self.Y.index[id_split], model + "opt_threshold_log_score"] = \
                            opt_threshold[model]["log_score"]
                            self.Y_hat.loc[self.Y.index[id_split], model + "opt_threshold_g_means"] = \
                            opt_threshold[model]["g_means"]
                            self.Y_hat.loc[self.Y.index[id_split], model + "opt_threshold_J"] = opt_threshold[model][
                                "J"]
                            self.Y_hat.loc[self.Y.index[id_split], model + "opt_threshold_f1_score"] = \
                            opt_threshold[model]["f1_score"]

                print(self.Y_hat.filter(["true_label", "logit_label"]).iloc[198:id_split + 10])


            else:

                print("predicting  Y_%i | X from 0 to %i, date:%s" % (id_split, id_split - self.step_ahead-1, self.Y.index[id_split].strftime('%Y-%m-%d')))
                Y_train_US = self.Y.loc[self.Y.index[0:id_split - self.step_ahead],"true_label"]
                X_test_US = self.X.iloc[id_split:, :]

                if method_2:
                    print('\t with method 2 \n')
                    for i in range(0,self.step_ahead):
                        Y_train_US[self.Y.index[id_split - self.step_ahead + i + 1]] = Y_train_US[id_split - self.step_ahead-1]
                    X_train_US = self.X.iloc[0:id_split, :]
                else:
                    X_train_US = self.X.iloc[0:id_split - self.step_ahead, :]

                print(Y_train_US)
                if resample:

                    resampled = resample_dataframe(pd.concat([X_train_US, Y_train_US], axis=1))
                    Y_train_US = resampled.iloc[:, -1]
                    X_train_US = resampled.iloc[:, :-1]
                print(Y_train_US)
                # If True normalizes data

                if normalize:

                    # Here we can also use standardization_norm()

                    X_train_US, X_test_US = standardization_norm(X_train_US, X_test_US)

                    # Drop nan observations generated by normalization

                    if X_train_US.isna().any().any():
                        aux = pd.concat([X_train_US, Y_train_US], axis=1).dropna()
                        Y_train_US = aux.iloc[:, -1]
                        X_train_US = aux.iloc[:, :-1]

                if params_tuning and id_split == len(self.X) - 1:
                    params_grid_logit = {'penalty': ['l1', 'l2'],
                                         'C': [0.01,0.1, 1.0, 10.0,100],
                                         'fit_intercept': [True, False],
                                         'solver': ['liblinear', 'saga']}

                    params_grid_gb = {'learning_rate': [0.05, 0.1, 0.2],
                                              'n_estimators': [50, 100, 200],
                                              'max_depth': [3, 4, 5],
                                              # 'min_samples_split': [2, 4, 6],
                                              # 'subsample': [0.6, 0.8, 1.0],
                                              'max_features': ['sqrt', 'log2', None]}

                    params_grid_rf = {'n_estimators': [50, 100, 200],
                                      # 'max_depth': [None, 10, 20, 30],
                                      # 'min_samples_split': [2, 5, 10],
                                      # 'min_samples_leaf': [1, 2, 4],
                                      'max_features': ['sqrt', 'log2', None]}
                    print('\t Params tuning for Logit \n')
                    grid_search_cv_logit = GridSearchCV(LogisticRegression(max_iter=5000,random_state=42),
                                                        params_grid_logit,
                                                        scoring="f1", cv=5)
                    print('\t Params tuning for GB \n')

                    grid_search_cv_gb = GridSearchCV(XGBClassifier(random_state=42),
                                                     params_grid_gb,
                                                     scoring="neg_log_loss", cv=5)

                    print('\t Params tuning for RF \n')

                    grid_search_cv_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                                                     params_grid_rf,
                                                     scoring="neg_log_loss", cv=5)

                    model_logit = grid_search_cv_logit.fit(X_train_US, Y_train_US)
                    print("\t\t",model_logit.best_params_,"\n")

                    model_gb = grid_search_cv_gb.fit(X_train_US, Y_train_US)
                    print("\t\t",model_gb.best_params_,"\n")

                    model_rf = grid_search_cv_rf.fit(X_train_US, Y_train_US)
                    print("\t\t",model_rf.best_params_,"\n")


                    models = {"logit": model_logit, "rf": model_rf, "gb": model_gb}


                else:

                    print('\t Logit \n')
                    model_logit = LogisticRegression(max_iter=5000, random_state=42, C=1).fit(X_train_US, Y_train_US)

                    print('\t RF \n')
                    model_rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train_US, Y_train_US)

                    print('\t GB \n')
                    model_gb = XGBClassifier(n_estimators=200, learning_rate=0.1,
                                             random_state=42).fit(X_train_US, Y_train_US)


                    models = {"logit_": model_logit, "rf_": model_rf, "gb_": model_gb}

                if threshold_tuning:

                    print("\t \t threshold tuning for the %i th observation" % id_split)

                    opt_threshold = get_optimal_thresholds(models, X_train_US,Y_train_US)

                models = {"logit_":model_logit,"rf_":model_rf,"gb_":model_gb}

                for (model_name,model) in zip(models,models.values()):
                    p = '{:.2f}'.format(model.predict_proba(X_test_US)[0][1])
                    self.Y_hat.loc[self.Y.index[id_split],model_name + "probs"] = p
                    self.Y_hat.loc[self.Y.index[id_split],model_name + "label"] = model.predict(X_test_US)[0]

                    if threshold_tuning:
                        self.Y_hat.loc[self.Y.index[id_split], model_name + 'opt_label_log_score'] = \
                            (lambda x: 1 if float(p) > x else 0)(opt_threshold[model_name]["log_score"])
                        self.Y_hat.loc[self.Y.index[id_split], model_name + 'opt_label_log_score'] = \
                            (lambda x: 1 if float(p) > x else 0)(opt_threshold[model_name]["log_score"])
                        self.Y_hat.loc[self.Y.index[id_split], model_name + 'opt_label_g_means'] = \
                            (lambda x: 1 if float(p) > x else 0)(opt_threshold[model_name]["g_means"])
                        self.Y_hat.loc[self.Y.index[id_split], model_name + 'opt_label_J'] = \
                            (lambda x: 1 if float(p) > x else 0)(opt_threshold[model_name]["J"])
                        self.Y_hat.loc[self.Y.index[id_split], model_name + 'opt_label_f1_score'] = \
                            (lambda x: 1 if float(p) > x else 0)(opt_threshold[model_name]["f1_score"])




                if threshold_tuning:
                    for model in ["logit_", "rf_", "gb_"]:
                        self.Y_hat.loc[self.Y.index[id_split],model + "opt_threshold_log_score"]= '{:.2f}'.format(opt_threshold[model][
                            "log_score"])
                        self.Y_hat.loc[self.Y.index[id_split],model + "opt_threshold_g_means"] = '{:.2f}'.format(opt_threshold[model][
                            "g_means"])
                        self.Y_hat.loc[self.Y.index[id_split],model + "opt_threshold_J"]= '{:.2f}'.format(opt_threshold[model]["J"])
                        self.Y_hat.loc[self.Y.index[id_split],model + "opt_threshold_f1_score"] = '{:.2f}'.format(opt_threshold[model][
                            "f1_score"])

        importance_logit = model_logit.coef_[0]
        importance_rf = model_rf.feature_importances_
        importance_gb = model_gb.feature_importances_

        feature_importance = {"logit_": importance_logit,"rf_": importance_rf,"gb_": importance_gb}

        if pca:
            return models, self.Y_hat, most_imp[-1]
        else:
            return models, self.Y_hat, self.acc_f_time, feature_importance
