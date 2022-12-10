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
X_lagged = data.lagged_covariates()
# Y = data.lagged_target()
# print(X_lagged)
# pprint(Y)
models = Models(Y=data.target(), X=X_lagged, date_split=204, step_ahead=15)
models.predict()
print(models.Y_test_label)
print(models.Y_test_label.dropna())
models.plot()
models.show_confusion_matrix()
# pprint(x)

#todo : corriger les indices
