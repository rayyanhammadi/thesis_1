from Main.Data_processing import Data, rebalance_dataframe, minmax_norm
from Main.Modelization import Models
from pprint import pprint
import pandas as pd
import math
import plotly as plt

from matplotlib import pyplot as plt

from pandas import value_counts
#test_git
# pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
BDD_path = "./Data/BDD_SummerSchool_BENOIT.xlsx"
BDD_sheet = "raw_data"
data = Data(BDD_path, BDD_sheet)
data.data_processing()
a,b=minmax_norm(data.covariates().iloc[204:300,:],data.covariates().iloc[300:,:])
print(a,b.iloc[0])
