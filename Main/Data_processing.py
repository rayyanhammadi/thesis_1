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
import statsmodels.api as sm
import pywt

from os.path import exists

def risk_free_index_processing():
    history_13w_ustb = yf.YahooFinancials('^IRX').get_historical_price_data('2002-01-01', '2020-12-31', 'monthly')
    history_10y_ustb = yf.YahooFinancials('^TNX').get_historical_price_data('2002-01-01', '2020-12-31', 'monthly')

    df1 = pd.DataFrame(history_13w_ustb['^IRX']['prices'])
    df2 = pd.DataFrame(history_10y_ustb['^TNX']['prices'])
    df1.drop('date', axis=1, inplace=True)
    df2.drop('date', axis=1, inplace=True)

    df1.index = pd.to_datetime(df1['formatted_date'])
    df2.index = pd.to_datetime(df2['formatted_date'])

    df1["price"] = df1["adjclose"]
    df2["price"] = df2["adjclose"]

    df1 = df1.filter(["price"])
    df2 = df2.filter(["price"])

    # df = df.resample('D').ffill()
    df1 = df1.resample('D').mean()  # Resample to daily frequency and aggregate using mean
    df2 = df2.resample('D').mean()
    # df = df.resample('D').ffill()
    df1 = df1.interpolate()
    df2 = df2.interpolate()# Interpolate missing values using linear interpolation
    df1 = df1[df1.index.day == 15]
    df2 = df2[df2.index.day == 15]
    # filename = "sp500_historical_data.txt"
    # df.to_csv(filename, sep='\t', index=True)
    return df1,df2

def risky_index_processing():
    history_sp = yf.YahooFinancials('^GSPC').get_historical_price_data('2002-01-01', '2020-12-31', 'monthly')
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
def resample_dataframe(df,over=True,under=True):
    # Count the number of labels in the dataframe
    label_counts = df['USA (Acc_Slow)'].value_counts()
    #print(label_counts)
    # Determine the minority and majority classes
    minority_label = label_counts.idxmin()
    majority_label = label_counts.idxmax()

    if over:

        # Determine the number of samples to keep from the minority class

        majority_count = label_counts[majority_label]

        # Sample the minority class
        minority_df = df[df['USA (Acc_Slow)'] == minority_label].sample(n=majority_count, replace=True,random_state=42)
    else:
        minority_df = df[df['USA (Acc_Slow)'] == minority_label]

    if under:
        minority_count = label_counts[minority_label]

        #  Sample the majority class
        majority_df = df[df['USA (Acc_Slow)'] == majority_label].sample(n=minority_count, replace=True,random_state=42)

    else:
        majority_df = df[df['USA (Acc_Slow)'] == minority_label]

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

    # Normalize the data in X_train and X_test using the trained scaler
    X_train_normalized = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_normalized = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

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

    def covariates(self,returns=False, log=False, ts_analysis=False, wavelet=False, diff=False):
        if returns:
            aux = self.df.iloc[:, :-1].loc[:, (self.df.iloc[:, :-1] > 0).all()]

            df_returns = aux.pct_change()

            # Rename the columns of the returns DataFrame to match the original columns
            if log :
                aux = self.df.iloc[:,:-1].loc[:, (self.df.iloc[:,:-1] > 0).all()]

                df_returns = np.log(aux.iloc[:, 1:]) - np.log(aux.iloc[:, 1:].shift(1))

                df_returns.columns = [column + 'log_change' for column in df_returns.columns]
            else:
                df_returns.columns = [column + '_change' for column in df_returns.columns]

            new_dataset = pd.concat([self.df.iloc[:, :-1], df_returns], axis=1)

            # Replace the first row (became NaN due to .pct()) by interpolation
            new_dataset.iloc[0] = new_dataset.iloc[1]


            return new_dataset

        elif ts_analysis:
            cov = self.df.iloc[:, :-1].copy()
            cov[cov.columns + '_rolling_avg'] = cov[cov.columns].rolling(window=3).mean()
            cov.fillna(method="bfill",inplace=True)

            # Add a new column with the seasonal decomposition of the original column
            for col in cov.columns:
                decomposition = sm.tsa.seasonal_decompose(cov[col], model='additive', period=3)
                cov[col+ '_trend'] = decomposition.trend
                cov[col+ '_seasonal'] = decomposition.seasonal
                cov[col + '_residual'] = decomposition.resid
                cov.fillna(method="bfill",inplace=True)
                cov.fillna(method="ffill", inplace=True)

            return cov

        elif wavelet:

            cov = self.df.iloc[:, :-1].copy()

            # Define the wavelet to be used
            wavelet = 'db4'

            # Loop over each column in the dataframe
            for col in cov.columns:
                # Perform the wavelet transform
                coeffs = pywt.wavedec(cov[col], wavelet)

                # Reconstruct the signal from the wavelet coefficients
                reconstructed = pywt.waverec(coeffs, wavelet)

                # Add the reconstructed signal to the original dataframe as a new column
                cov[f"{col}_reconstructed"] = reconstructed

            return cov
        elif diff:

            cov = self.df.iloc[:, :-1].copy()

            for col in cov.columns:
                for i in range(36):
                    cov[f"{col}_diff_{i}"] = cov[col].diff(periods=i)

            cov.fillna(method="bfill", inplace=True)

            return cov



        else:

            return self.df.iloc[:,:-1]


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


