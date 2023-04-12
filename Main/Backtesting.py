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
        self.rf_rate = risk_free_index[0]
        self.bond_rate = risk_free_index[1]
        self.risky_assets = risky_index
        self.strategy = strategy

    def simulation(self):
        n_risky_assets_held = 0
        current_cash = self.capital

        # Portfolio value starts with current cash
        self.portfolio_history["portfolio_value"].iloc[0] = current_cash


        if self.strategy == "dynamic":
            for date in range(1,len(self.y_pred)-1):
                print(date)
                print(current_cash)
                print(n_risky_assets_held)

                # Bond return in %
                bond_return = (1+self.bond_rate['price'].iloc[date])**(1/12)-1

                # When we have cash placed in riskfree asset
                if current_cash > 0 and date > 1:
                    # Add monthly gains from riskfree cash allocation
                    current_cash += current_cash * bond_return/100

                    # Model predicts acceleration
                    if self.y_pred.iloc[date] == 1:

                        if date > 0:
                            # Sell last portfolio

                            current_cash += n_risky_assets_held * self.risky_assets['price'].iloc[date]

                        # Buy new portfolio

                        # Allocate 80% of cash in risky asset

                        n_risky_assets_held = (current_cash * .8)/self.risky_assets['price'].iloc[date]

                        # Allocate 20% of cash in risky asset

                        current_cash = .2 * current_cash

                    else:

                        # Sell last portfolio
                        if date>0:
                            current_cash += n_risky_assets_held * self.risky_assets['price'].iloc[date]

                        # Allocate 60% of cash in risky asset
                        n_risky_assets_held = (current_cash * .4) / self.risky_assets['price'].iloc[date]

                        # Allocate 40% of cash in risky asset
                        current_cash = .6 * current_cash


                # Evaluate and store portfolio value each month



                self.portfolio_history["portfolio_value"].iloc[date] = n_risky_assets_held * \
                                                                       self.risky_assets['price'].iloc[date] \
                                                                       + current_cash

            # Compute portfolio returns and returns in %
            self.portfolio_history["return_pct"] = self.portfolio_history["portfolio_value"].pct_change()

        elif self.strategy == "120/80_equity":
            # Number of months we were in acceleration phase
            # To calculate cost of leverage
            n_months = 0

            # To track leverage cost
            borrowed_cash = 0


            for date in range(1,len(self.y_pred)-1):

                borrowing_cost = ((1+self.rf_rate['price'].iloc[date])**(1/12)-1 )/ 100

                # Model predicts acceleration
                if self.y_pred.iloc[date] == 1:

                    n_months += 1

                    # Sell last portfolio
                    if date>0:
                        current_cash += n_risky_assets_held * self.risky_assets['price'].iloc[date]

                    # Buy new portfolio

                    # Allocate 120% of cash in risky asset
                    n_risky_assets_held = (current_cash * 1.2)/self.risky_assets['price'].iloc[date]

                    #Borrow 20%
                    borrowed_cash = 0.2 * current_cash
                    current_cash = - borrowed_cash


                else:
                    n_months=0

                    if date>0:
                        if current_cash!=0 and borrowed_cash>0:
                            current_cash -= borrowed_cash * n_months * borrowing_cost

                    # Sell last portfolio

                    current_cash += n_risky_assets_held * self.risky_assets['price'].iloc[date]

                    # Allocate 80% of cash in risky asset

                    n_risky_assets_held = (current_cash * .6) / self.risky_assets['price'].iloc[date]

                    # Allocate 20% of cash in riskfree asset

                    current_cash = .4 * current_cash




                # Evaluate and store portfolio value in each iteration


                self.portfolio_history["portfolio_value"].iloc[date] = n_risky_assets_held * \
                                                                       self.risky_assets['price'].iloc[date] \
                                                                       + current_cash

            # Compute portfolio returns in %

            self.portfolio_history["return_pct"] = self.portfolio_history["portfolio_value"].pct_change()

        return self.portfolio_history["portfolio_value"], self.portfolio_history["return_pct"]

    def backtest_report(self):
        if self.strategy == "dynamic":
            print("************* Descriptive Statistics for the Dynamic Strategy *************")
        else:
            print("************* Descriptive Statistics for the 120/80  Strategy *************")
        print("Period", len(self.y_pred), "days")
        print("Max Monthly Drawdown", 100 * round(self.portfolio_history["return_pct"].min(), 2), "%")
        print("Max Monthly Drawdown BH", 100 * round(self.risky_assets['price'].pct_change().min(), 2), "%")

        print("Highest Monthly Return ", 100 * round(self.portfolio_history["return_pct"].max(), 2), "%")
        print("Average  Returns ", 100 * self.portfolio_history["return_pct"].mean(), "%")
        print("Average  Returns of the becnhmark ", 100 * self.risky_assets['price'].pct_change().mean(), "%")

        print("Volatility", 100 * round(self.portfolio_history["return_pct"].std(), 2), "%")
        print("Total Potential Return ", 100 * (round(sum(np.where((self.portfolio_history["return_pct"]> 0), self.portfolio_history["return_pct"], 0)), 2)),
              "%")
        print("Total Potential Loss ", 100 * (round(sum(np.where((self.portfolio_history["return_pct"] < 0), self.portfolio_history["return_pct"], 0)), 2)),
              "%")
        print("Net Return ", 100 * self.portfolio_history["return_pct"].sum().round(2), "%")
        print("Sharpe ratio", compute_sharpe_ratio(self.portfolio_history["return_pct"]))
        print("**************************************************")

    def plots(self):
        # Plots
        df = pd.concat([self.portfolio_history['portfolio_value'], self.y_pred], axis=1)

        f, axarr = plt.subplots(2, figsize=(12, 7))
        if self.strategy == "120/80_equity":
            f.suptitle('Portfolio Value and Return with the 120/80 Equity strategy', fontsize=20)
        else:
            f.suptitle('Portfolio Value and Return with the dynamic strategy', fontsize=20)


        axarr[0].plot(self.portfolio_history["portfolio_value"], color='blue')
        axarr[0].plot(self.risky_assets['price'], color='black')
        axarr[0].scatter(df[df["label"] == 1].index,
                         df["portfolio_value"][df["label"] == 1], color='green', marker='.', s=100,
                         label="Predicted Acc.")
        axarr[0].scatter(df[df["label"] == 0].index,
                         df["portfolio_value"][df["label"] == 0], color='red', marker='.', s=100,
                         label="Predicted Slo.")

        axarr[0].grid(True)
        axarr[1].plot(100 * self.portfolio_history["return_pct"], color='red')
        axarr[1].grid(True)
        f.legend(['Portfolio value', 'B&H SP500', "Predicted Acc.", "Predicted Slo.", 'Return'], loc='upper left')
        plt.show()