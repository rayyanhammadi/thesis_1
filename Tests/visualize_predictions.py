from Main.Data_processing import Data
from Main.Modelization import Models
from Main.Backtesting import Portfolio
from utils.helpers import plot_predictions, show_confusion_matrix, read_predictions, compare_models
from pprint import pprint
from matplotlib import pyplot as plt
import pandas as pd
from os.path import exists

pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
BDD_path = "../Data/BDD_SummerSchool_BENOIT.xlsx"
BDD_sheet ="raw_data"
data = Data(BDD_path, BDD_sheet)
data.data_processing()

X=data.covariates()
Y = data.target()

# If available reads stored predictions

f = open("normalisation_meth1_with_lag_c1_rf200_gb200.txt",'r')
g = open("normalisation_meth1_over_under_sampling_with_lag_c1_rf200_gb200.txt",'r')
y_hat = read_predictions(f)
y_hat_ = read_predictions(g)

plot, compare= True, True

if plot:
    pred_probs = [y_hat["logit_probs"].iloc[204:430],y_hat["rf_probs"].iloc[204:430],y_hat["gb_probs"].iloc[204:430]]
    pred_labels = [y_hat["logit_label"].iloc[204:430],y_hat["rf_label"].iloc[204:430],y_hat["gb_label"].iloc[204:430]]
    opt_threshold = [y_hat["logit_opt_threshold_f1_score"].iloc[204:430],y_hat["rf_opt_threshold_f1_score"].iloc[204:430],y_hat["gb_opt_threshold_f1_score"].iloc[204:430]]

    pred_probs_ = [y_hat_["logit_probs"].iloc[204:430], y_hat_["rf_probs"].iloc[204:430], y_hat_["gb_probs"].iloc[204:430]]
    pred_labels_ = [y_hat_["logit_label"].iloc[204:430], y_hat_["rf_label"].iloc[204:430], y_hat_["gb_label"].iloc[204:430]]
    opt_threshold_ = [y_hat_["logit_opt_threshold_f1_score"].iloc[204:430],
                     y_hat_["rf_opt_threshold_f1_score"].iloc[204:430], y_hat_["gb_opt_threshold_f1_score"].iloc[204:430]]
    names = ["Logistic Regression","Random Forest Classifier","XGradientBoosting Classifier"]

    plot_predictions(y_label=data.target().iloc[204:],y_probs=pred_probs,names=names)
    show_confusion_matrix(labels=data.target().iloc[204:430],preds=pred_labels, names=names)

    plot_predictions(y_label=data.target().iloc[204:], y_probs=pred_probs_, names=names)
    show_confusion_matrix(labels=data.target().iloc[204:430], preds=pred_labels_, names=names)

#compare models
if compare:
    models = ["logit_","rf_","gb_"]
    for model_name in models:
        print(model_name)
        compare_models(y_true=data.target().iloc[204:430],y_model2=y_hat_[model_name + "label"].iloc[204:430],
                   y_model1=y_hat[model_name + "label"].iloc[204:430])





