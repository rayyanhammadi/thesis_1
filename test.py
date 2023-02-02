from Main.Data_processing import Data
from Main.Modelization import Models
from pprint import pprint
import pandas as pd
import math

from matplotlib import pyplot as plt

from pandas import value_counts
#test_git
# pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
BDD_path = "./Data/BDD_SummerSchool_BENOIT.xlsx"
BDD_sheet = "raw_data"
data = Data(BDD_path, BDD_sheet)
data.data_processing()
# X1=data.PCA(data.standardization_norm(data.covariates()),.99)
# Y = data.target()
print(data.data_summary())
# Y['USA (Acc_Slow)'] = math.floor(Y[:,1])

data.stationarity()
# todo : corriger les indices
