from Main.Data_processing import Data
from Main.Modelization import Models
import pandas as pd
#test_git

BDD_path = "./Data/BDD_SummerSchool_BENOIT.xlsx"
BDD_sheet = "raw_data"
data = Data(BDD_path, BDD_sheet)
data.data_processing()
models = Models(data.df, 204, 15, 0)
models.split_2()
models.fit()
models.predict()
#print(models.Y_test_US)
#print(models.Y_test_US_1)
print(models.Y_test_US_1)
print(models.plot())
print(models.show_matrix_confusion())