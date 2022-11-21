import math
import numpy as np
import pandas as pd

class Data:
    def __init__(self,BDD_path, BDD_sheet):
        self.raw_df = pd.read_excel(BDD_path, sheet_name=BDD_sheet)
        self.df = None

    def data_processing(self):
        self.df = self.raw_df.iloc[1:,:]
        self.df.columns = self.raw_df.iloc[0,:]
        self.df.set_index("dates", drop=True, inplace=True)
        self.df = self.df.astype("float")
        print("Data processed succesfully")
        return(self.df)

    def data_summary(self):
        print(self.df.head())
        print(self.df.describe())



