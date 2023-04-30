from Main.Data_processing import Data, risky_index_processing, risk_free_index_processing
from Main.Modelization import Models
from Main.Backtesting import Portfolio
from utils.helpers import plot_predictions, show_confusion_matrix, store_predictions, read_predictions
from pprint import pprint
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

f = open("normalisation_meth1_over_under_sampling_with_lag_c1_rf200_gb200.txt", "r")
g = open("normalisation_resample_over_under_with_lag_c1_rf200_gb200.txt","r")
h = open("normalisation_with_lag_c1_rf200_gb200.txt","r")
y_hat= read_predictions(f)
y_hat_ = read_predictions(g)
y_hat__ = read_predictions(h)


pred_labels = [y_hat["rf_label"].iloc[204:430], y_hat_["gb_label"].iloc[204:430], y_hat__["logit_label"].iloc[204:430]]
names = ["Random Forest Classifier", "XGradientBoosting Classifier","Logistic Regression"]

# Create portfolio
portfolio = Portfolio(initial_capital=1146.18994140625,risky_index=risky_index_processing(),
                      risk_free_index=risk_free_index_processing(),y_pred=pred_labels[0],strategy="dynamic")

portfolio_ = Portfolio(initial_capital=1146.18994140625,risky_index=risky_index_processing(),
                      risk_free_index=risk_free_index_processing(),y_pred=pred_labels[1],strategy="dynamic")

portfolio__ = Portfolio(initial_capital=1146.18994140625,risky_index=risky_index_processing(),
                      risk_free_index=risk_free_index_processing(),y_pred=pred_labels[2],strategy="dynamic")

# Simulation of the strategy and plots
portfolio.simulation()
portfolio_.simulation()
portfolio__.simulation()

portfolios_history = [portfolio.portfolio_history,portfolio_.portfolio_history,portfolio__.portfolio_history]

portfolio.plots(portfolios_history=portfolios_history,y_preds=pred_labels,names=names)

# Backtest report
portfolio.backtest_report(portfolios_history=portfolios_history,names=names)



