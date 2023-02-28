from Main.Data_processing import Data
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

# Creates model
model = Models(name="RF",Y=data.target(), X=X, date_split=204, step_ahead=12)

# Run model
y_hat=model.predict()[1]

# If available reads stored predictions

#f = open("logit_pro_max_predictions.txt",'r')
#y_hat = read_predictions(f)


#If not stores prediction
store_predictions(y_hat,"RF_pro_max")


plot_predictions(y_label=data.target().iloc[204:],y_probs=y_hat["probs"],name="GB_pro_max")
show_confusion_matrix(labels=data.target().iloc[205:431],preds=y_hat["label"].iloc[205:431], name="GB_pro_max")




