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
X_lagged_plus = data.lagged_covariates_plus()
model_3 = Models(name="lagged + targetlagged", Y=data.target(), X=X_lagged_plus, date_split=204, step_ahead=12)
model_3.ew_predict()
model_3.plot()
model_3.show_confusion_matrix()

# todo : corriger les indices
