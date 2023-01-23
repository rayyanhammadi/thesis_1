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
X1 = data.standardization_norm()
X2 = data.minmax_norm()
X3 = data.robust_norm()

Y = data.target()
model_1 = Models(name="ABC",Y=data.target(), X=X1, date_split=204, step_ahead=15)
model_2 = Models(name="logit",Y=data.target(), X=X2, date_split=204, step_ahead=15)
model_3 = Models(name="3",Y=data.target(), X=X3, date_split=204, step_ahead=15)

model_1.predict()


model_1.plot()

# print(model_1.Y_test_label)
model_1.show_confusion_matrix()



# todo : corriger les indices
