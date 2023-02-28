import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils.helpers import compute_sharpe_ratio


class Portfolio:

    def __init__(self, initial_capital:float,risk_free_index=None,risky_index=None,strategy=None, y_pred=None):

        self.capital = initial_capital
        self.y_pred = y_pred
        self.portfolio_history = pd.concat([pd.DataFrame(np.nan,index=y_pred.index, columns=["portfolio_value"]),
                                            pd.DataFrame(np.nan,index=y_pred.index, columns=["portfolio_returns"]),
                                            pd.DataFrame(np.nan,index=y_pred.index, columns=["B&H_strategy"])], axis=1)
        self.rf_assets=risk_free_index
        self.risky_assets = risky_index
        self.strategy = strategy

    def simulation(self):
        n_risky_assets_held = 0
        current_cash = self.capital

        # Portfolio value starts with current cash
        self.portfolio_history["portfolio_value"].iloc[0] = current_cash
        if self.strategy == "dynamic":
            for date in range(1,len(self.y_pred)):
                print(date)
                print(current_cash)
                print(n_risky_assets_held)

                # If month is predicted acceleration
                if self.y_pred.iloc[date] == 1:

                    # If month is predicted slowdown

                    if self.y_pred.iloc[date-1] == 0:

                        # If has 0 risky asset

                        if n_risky_assets_held ==0:

                            # Allocate 80% of cash in risky asset

                            n_risky_assets_held = (current_cash * .8) /self.risky_assets['price'].iloc[date]

                            # Allocate 20% of cash in riskless asset
                            current_cash = .2 * current_cash

                        #

                        else:

                            # Sell last portfolio

                            current_cash += n_risky_assets_held * self.risky_assets['price'].iloc[date]

                            # Buy new portfolio

                            n_risky_assets_held = (current_cash * .8)/self.risky_assets['price'].iloc[date]
                            current_cash = .2 * current_cash

                else:
                    if self.y_pred.iloc[date-1] == 1:
                        if n_risky_assets_held == 0:
                            n_risky_assets_held = (current_cash * .4) / self.risky_assets['price'].iloc[date]
                            current_cash = .6 * current_cash

                        else:
                            current_cash += n_risky_assets_held * self.risky_assets['price'].iloc[date]
                            n_risky_assets_held = (current_cash * .4) / self.risky_assets['price'].iloc[date]
                            current_cash = .6 * current_cash

                # Evaluate and store portfolio value in each iteration


                self.portfolio_history["portfolio_value"].iloc[date] = n_risky_assets_held * \
                                                                       self.risky_assets['price'].iloc[date] \
                                                                       + current_cash

            # Compute portfolio returns and returns in %

            self.portfolio_history["return"] = self.portfolio_history["portfolio_value"].diff()
            self.portfolio_history["return_pct"] = self.portfolio_history["portfolio_value"].pct_change()

            #Plots
            df = pd.concat([self.portfolio_history['portfolio_value'],self.y_pred],axis=1)

            f, axarr = plt.subplots(2, figsize=(12, 7))
            f.suptitle('Portfolio Value and Return', fontsize=20)
            axarr[0].plot(self.portfolio_history["portfolio_value"], color='blue')
            axarr[0].plot(self.risky_assets['price'], color='black')
            axarr[0].scatter(df[df["label"]== 1].index,
                        df["portfolio_value"][df["label"]==1], color='green', marker='.',s=100, label="Predicted Acc.")
            axarr[0].scatter(df[df["label"]== 0].index,
                        df["portfolio_value"][df["label"]==0], color='red', marker='.',s=100, label="Predicted Slo.")

            axarr[0].grid(True)
            axarr[1].plot(self.portfolio_history["return_pct"] , color='red')
            axarr[1].grid(True)
            f.legend(['Portfolio value','B&H SP500',"Predicted Acc.","Predicted Slo.",'Return'], loc='upper left')
            plt.show()
        return self.portfolio_history["portfolio_value"], self.portfolio_history["return_pct"]

    def backtest_report(self):
        print("************* Descriptive Statistics *************")
        print("Period", len(self.y_pred), "days")
        print("Highest Monthly Loss ", 100 * round(self.portfolio_history["return_pct"].min(), 2), "%")
        print("Highest Monthly Return ", 100 * round(self.portfolio_history["return_pct"].max(), 2), "%")
        print("Average  Return ", self.portfolio_history["return_pct"].mean(), "%")
        print("Standard Deviation of Return ", 100 * round(self.portfolio_history["return_pct"].std(), 2), "%")
        print("Total Potential Return ", 100 * (round(sum(np.where((self.portfolio_history["return_pct"]> 0), self.portfolio_history["return_pct"], 0)), 2)),
              "%")
        print("Total Potential Loss ", 100 * (round(sum(np.where((self.portfolio_history["return_pct"] < 0), self.portfolio_history["return_pct"], 0)), 2)),
              "%")
        print("Net Return ", 100 * self.portfolio_history["return_pct"].sum().round(2), "%")
        print("Sharpe ratio", compute_sharpe_ratio(self.portfolio_history["return_pct"]))
        print("**************************************************")