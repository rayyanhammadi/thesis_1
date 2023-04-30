from Main.Data_processing import Data
from Main.Modelization import Models
from utils.helpers import store_predictions

import pandas as pd

#pd.set_option('display.max_columns',None)

# Change path here

BDD_path = "../Data/BDD_SummerSchool_BENOIT.xlsx"

# Change sheet name here

BDD_sheet = "raw_data"

# Process data
data = Data(BDD_path, BDD_sheet)
data.data_processing()

X = data.covariates()
Y = data.target()

# Change parameters here

tuning = {"normalize": True, "resample": True, "threshold_tuning": False, "pca": False, "params_tuning": False}
imputation_method = {"method_1":True, "method_2":False}

# Creates model
model = Models(Y=data.target(), X=X, date_split=204, step_ahead=12, tuning=tuning, method=imputation_method)

model_ = model.models()
models , y_hat, importance, acc = model_[0], model_[1], model_[-1], model_[-2]

# Store predictions
store_predictions(y_hat,"normalisation_meth1_over_under_sampling_with_lag_c1_rf200_gb200")





