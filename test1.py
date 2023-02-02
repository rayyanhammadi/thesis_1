from Main.Data_processing import Data
from Main.Modelization import Models
from pprint import pprint
import pandas as pd
#test_git
pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
BDD_path ="./Data/BDD_SummerSchool_BENOIT.xlsx"
BDD_sheet ="raw_data"
data = Data(BDD_path, BDD_sheet)
data.data_processing()

col_forward=["var51","var7","var41","var22","spread105","var34","var53","var55","var26","var3","var30","var42","spread13m","var31","var19","var1","var15","spi","var11","var35","spread102","var54","var49","gold","var43","var12","var32"]
col_both =["var51","var7","var41","var22","var53","var55","var26","var3","spread13m","var31","var1","var15","spi","var11","var35","spread102","var43","var49","gold","var9","var32","var20","var56"]

X1=data.PCA(data.standardization_norm(data.covariates()),.99)
Y = data.target()
model_1 = Models(name="BC",Y=data.target(), X=X1, date_split=204, step_ahead=12)
# model_2 = Models(name="RF",Y=data.target(), X=X2, date_split=204, step_ahead=15)
#
model_1.predict()
#
# model_2.predict()

model_1.plot()
#
# model_2.plot()

# print(model_1.Y_test_label)
model_1.show_confusion_matrix()

# model_2.show_confusion_matrix()



# todo : corriger les indices
