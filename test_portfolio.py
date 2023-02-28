from Main.Data_processing import Data, risky_index_processing
from Main.Modelization import Models
from Main.Backtesting import Portfolio
from utils.helpers import plot_predictions, show_confusion_matrix, store_predictions, read_predictions
from pprint import pprint
import pandas as pd
from os.path import exists

pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
BDD_path ="./Data/BDD_SummerSchool_BENOIT.xlsx"
BDD_sheet ="raw_data"
data = Data(BDD_path, BDD_sheet)
data.data_processing()

X=data.covariates()
Y = data.target()

model = Models(name="logit",Y=data.target(), X=X, date_split=204, step_ahead=12)
f = open("GB_pro_max_predictions.txt", "r")
y_hat= read_predictions(f)

# Create portfolio
portfolio = Portfolio(initial_capital=1146.18994140625,risky_index=risky_index_processing(),y_pred=y_hat["label"].iloc[204:],strategy="dynamic")

# Simulation of the strategy and plots

portfolio.simulation()

# Backtest report
portfolio.backtest_report()



