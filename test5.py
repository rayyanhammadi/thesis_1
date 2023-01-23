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
X_lagged = data.lag_covariates(data.minmax_norm())
model_5 = Models(name ="minmax+lagged", Y=data.target(), X=X_lagged, date_split=205, step_ahead=15)
model_5.ew_predict()
model_5.plot()
model_5.show_confusion_matrix()

# todo : corriger les indices
