import math
import numpy as np
import pandas as pd
from pprint import pprint

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

    def target(self):
        return self.df.iloc[:,-1]

    def lagged_target(self):
        return self.lag_target(self.df.iloc[:,-1])

    def covariates(self):
        return self.df.iloc[:,:-1]

    def lagged_covariates(self):
        return pd.concat([self.lag_covariates(self.df.iloc[:,:-1]),self.lagged_target()],axis=1)
    @staticmethod
    def lag_covariates(data, lag=18):
        for column in data:
            for i in range(18):
                data['%s_lag_%i' % (column, i+1)] = data[column].shift(i+1)
        return data.dropna()

    @staticmethod
    def lag_target(data, lag=5):
        y = pd.DataFrame(data)
        for i in range(18):
            y['Y_lag_%i' % (i+1)] = y["USA (Acc_Slow)"].shift(i+1)
        y = y.drop(columns=["USA (Acc_Slow)"])
        return y.dropna()

    def data_summary(self):
        print(self.df.head())
        # print(self.df.describe())



