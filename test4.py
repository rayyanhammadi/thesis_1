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
X_lagged = data.lag_covariates(data.standardization_norm())
model_4 = Models(name = "stand+lagged",Y=data.target(), X=X_lagged, date_split=204, step_ahead=12)
model_4.ew_predict()
model_4.plot()
model_4.show_confusion_matrix()

# todo : corriger les indices
