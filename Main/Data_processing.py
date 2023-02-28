import math
import numpy as np
import pandas as pd
from pprint import pprint
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import yahoofinancials as yf
from os.path import exists



def risky_index_processing():
    history_sp = yf.YahooFinancials('^GSPC').get_historical_price_data('2002-01-01', '2020-12-31', 'daily')
    df = pd.DataFrame(history_sp['^GSPC']['prices'])
    df.drop('date', axis=1, inplace=True)
    df.index = pd.to_datetime(df['formatted_date'])
    df["price"] = df["adjclose"]
    df = df.filter(["price"])
    df = df.resample('D').ffill()
    df = df[df.index.day == 15]
    filename = "sp500_historical_data.txt"
    df.to_csv(filename, sep='\t', index=True)
    return df
def resample_dataframe(df):
    # Count the number of labels in the dataframe
    label_counts = df['USA (Acc_Slow)'].value_counts()
    #print(label_counts)
    # Determine the minority and majority classes
    minority_label = label_counts.idxmin()
    majority_label = label_counts.idxmax()

    # Determine the number of samples to keep from the minority class
    minority_count = label_counts[minority_label]
    majority_count = label_counts[majority_label]
    #difference = majority_count - minority_count

    minority_to_keep = majority_count

    # Sample the minority class
    minority_df = df[df['USA (Acc_Slow)'] == minority_label].sample(n=700, replace=True,random_state=42)

    # Sample the majority class
    majority_df = df[df['USA (Acc_Slow)'] == majority_label].sample(n=700, replace=True,random_state=42)

    # Concatenate the minority and majority samples
    balanced_df = pd.concat([minority_df, majority_df])

    # Shuffle the samples
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df
def minmax_norm(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_normalized,X_test_normalized = pd.DataFrame(),pd.DataFrame()
    X_train_normalized[X_train.columns] = scaler.transform(X_train)
    X_test_normalized[X_test.columns] = scaler.transform(X_test)
    return X_train_normalized,X_test_normalized


def standardization_norm(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_normalized,X_test_normalized = pd.DataFrame(),pd.DataFrame()
    X_train_normalized[X_train.columns] = scaler.transform(X_train)
    X_test_normalized[X_test.columns] = scaler.transform(X_test)
    return X_train_normalized,X_test_normalized

def PCA_(data, important_features=False, n_comp=.99):
    """
    Effectue une PCA sur la matrice X

    :param data:
    :param important_features:
    :param n_comp:
    :return:
    """
    pca = PCA(n_components=n_comp)
    pca.fit_transform(data)
    n_pcs = pca.n_components_
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = data.columns
    most_important_features = [*set([initial_feature_names[most_important[i]] for i in range(n_pcs)])]
    if important_features:
        print(most_important_features)
    return most_important_features

class Data:
    def __init__(self,BDD_path, BDD_sheet):
        self.raw_df = pd.read_excel(BDD_path, sheet_name=BDD_sheet)
        self.df = None

    def data_processing(self,resample=False):
        """

        :param resample: Si vrai utilise une méthode de réchantillonage des données
        :return:
        """
        self.df = self.raw_df.iloc[1:,:]
        self.df.columns = self.raw_df.iloc[0,:]
        self.df.set_index("dates", drop=True, inplace=True)
        self.df = self.df.astype("float")
        if resample:
            self.df = self.df.resample('D',axis=0).interpolate('linear')

        print("Data processed succesfully")

    def target(self,resample=False):
        if resample:
            return self.df.iloc[:,-1].astype('int')
        return self.df.iloc[:,-1]

    def covariates(self):
        return self.df.iloc[:,:-1]

    def lagged_covariates(self):
        return self.lag_covariates(self.df.iloc[:,:-1])

    @staticmethod
    def lag_covariates(data, lag=18):
        for column in data:
            for i in range(0,18):
                data['%s_lag_%i' % (column, i+1)] = data[column].shift(i)
        return data.dropna()
    @staticmethod
    def covariates_w_returns(data):
        for column in data:
            for i in range(0,20):
                data['%s_diff_%i' % (column, i+1)] = data[column].diff(i)
        return data.dropna()

    def data_summary(self):

        print(self.df.head())
        print(self.df.describe())
        print(self.target().value_counts())

    def stationarity(self):
        """
        Check si la série temporelle Y est stationnaire ou non.
        :return: le résultat du test
        """

        print('Dickey-Fuller Test: H0 = non stationnaire vs H1 = stationnaire')
        dftest = adfuller(self.target(),autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=["Statistique de test","p-value","lags","nobs"])
        for key,value in dftest[4].items():
            dfoutput['valeur critique(%s)'%key]=value
        print(dfoutput)

    @staticmethod
    def minmax_norm(data):
        scaler = MinMaxScaler()
        df = pd.DataFrame()
        df[data.columns] = scaler.fit_transform(data)
        return df
    @staticmethod
    def normalize_norm(data):
        scaler = MinMaxScaler()
        df = pd.DataFrame()
        df[data.columns] = scaler.fit_transform(data)
        return df
    @staticmethod
    def robust_norm(data):
        scaler = RobustScaler()
        df = pd.DataFrame()
        df[data.columns] = scaler.fit_transform(data)
        return df
    @staticmethod
    def standardization_norm(data):
        scaler = StandardScaler()
        df = pd.DataFrame()
        df[data.columns] = scaler.fit_transform(data)
        return df
    @staticmethod
    def PCA_(data, important_features=False, n_comp=.99):
        """
        Effectue une PCA sur la matrice X

        :param data:
        :param important_features:
        :param n_comp:
        :return:
        """
        pca = PCA(n_components=n_comp)
        pca.fit_transform(data)
        n_pcs = pca.n_components_
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        initial_feature_names = data.columns
        most_important_features = [*set([initial_feature_names[most_important[i]] for i in range(n_pcs)])]
        if important_features:
            print(most_important_features)
        return most_important_features





#----------------Data Analysis-------------------

    @staticmethod
    def ts_decomposition(df, col_name='USA (Acc_Slow)', samples='all', period=12):
        """
        Plot la décomposition de la série temporelle Yt

        :param df:
        :param col_name:
        :param samples:
        :param period:
        :return:
        """
        if samples == 'all':
            # decomposing all time series timestamps
            res = seasonal_decompose(df[col_name].values, period=period)
        else:
            # decomposing a sample of the time series
            res = seasonal_decompose(df[col_name].values[-samples:], period=period)

        observed = res.observed
        trend = res.trend
        seasonal = res.seasonal
        residual = res.resid

        # plot the complete time series
        fig, axs = plt.subplots(4, figsize=(16, 8))
        axs[0].set_title('OBSERVED', fontsize=16)
        axs[0].plot(observed)
        axs[0].grid()

        # plot the trend of the time series
        axs[1].set_title('TREND', fontsize=16)
        axs[1].plot(trend)
        axs[1].grid()

        # plot the seasonality of the time series. Period=24 daily seasonality | Period=24*7 weekly seasonality.
        axs[2].set_title('SEASONALITY', fontsize=16)
        axs[2].plot(seasonal)
        axs[2].grid()

        # plot the noise of the time series
        axs[3].set_title('NOISE', fontsize=16)
        axs[3].plot(residual)
        axs[3].scatter(y=residual, x=range(len(residual)), alpha=0.5)
        axs[3].grid()

        plot_acf(df[col_name].values, lags=400)
        plt.show()


