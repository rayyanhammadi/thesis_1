from Main.Data_processing import Data
from Main.Modelization import Models
from pprint import pprint
import pandas as pd
#test_git
pd.set_option('display.max_column',None)
# pd.set_option('display.max_rows',None)
BDD_path = "./Data/BDD_SummerSchool_BENOIT.xlsx"
BDD_sheet = "raw_data"
data = Data(BDD_path, BDD_sheet)
data.data_processing()
X = data.covariates()
# print(X.iloc[0,:])
print(X[['spread63m', 'spread13m', 'spread23m',
       'spread53m', 'spread103m', 'spread102','spread105','spread52','spread21']])


# todo : corriger les indices
