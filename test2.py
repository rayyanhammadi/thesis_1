from Main.Data_processing import Data
from Main.Modelization import Models
from pprint import pprint
import pandas as pd
#test_git
pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
BDD_path = "./Data/BDD_SummerSchool_BENOIT.xlsx"
BDD_sheet = "raw_data"
data = Data(BDD_path, BDD_sheet)
data.data_processing()
Y = data.target()
X_lagged = data.minmax_norm()
model_2 = Models(name="lagged +ew",Y=data.target(), X=X_lagged, date_split=204, step_ahead=15)
model_2.logit_expanding_window_prediction_method()
model_2.plot()
model_2.show_confusion_matrix()

# todo : corriger les indices
