import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
from utils.helpers import compute_sharpe_ratio
from PIL import Image, ImageDraw, ImageFont


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

    def backtest_report_(self,portfolios_history: list = None, names: list = None):
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

    import pandas as pd
    from tabulate import tabulate

    import tabulate

    import pandas as pd
    from tabulate import tabulate

    import pandas as pd
    from tabulate import tabulate

    import pandas as pd
    from tabulate import tabulate

    def backtest_report(self, portfolios_history: list = None, names: list = None):
        """
        Generate a report for each portfolio in the input list.

        Args:
        portfolios_history (list): a list of dictionaries containing the history of each portfolio
        names (list): a list of names corresponding to each portfolio

        Returns:
        None
        """

        reports = []
        for portfolio_history, name in zip(portfolios_history, names):
            report = {}
            report['Period'] = len(portfolio_history)
            report['Max Monthly Drawdown in %'] = 100 * round(portfolio_history["return_pct"].min(), 2)
            report['Highest Monthly Return in %'] = 100 * round(portfolio_history["return_pct"].max(), 2)
            report['Average Returns in %'] = 100 * portfolio_history["return_pct"].mean()
            report['Volatility'] = 100 * round(portfolio_history["return_pct"].std(), 2)
            report['Net Return in %'] = 100 * portfolio_history["return_pct"].sum().round(2)
            report['Sharpe ratio'] = compute_sharpe_ratio(portfolio_history["return_pct"])
            reports.append(report)


        df = pd.DataFrame(reports)
        df.set_index('Period', inplace=True)
        df = df.transpose()
        df.columns = names
        print(tabulate(df, headers='keys', tablefmt='fancy_grid'))



    def plots(self, portfolios_history: list = None, y_preds: list = None, names: list = None):
        # Plots

        f, axarr = plt.subplots(2, figsize=(12, 7))
        if self.strategy == "120/80_equity":
            f.suptitle('Portfolio Value and Return with the 120/80 Equity strategy', fontsize=20)
        else:
            f.suptitle('Portfolio Value and Return with the dynamic strategy', fontsize=20)

        model_names = ["rf_", "gb_","logit_"]
        for (portfolio_history, y_pred, name, model_name) in zip(portfolios_history, y_preds, names, model_names):
            df = pd.concat([portfolio_history['portfolio_value'], y_pred], axis=1)

            axarr[0].plot(portfolio_history["portfolio_value"], label=f"{name} Portfolio", linewidth=2.5)
            axarr[0].scatter(df[df[model_name + "label"] == 1].index,
                             df["portfolio_value"][df[model_name + "label"] == 1], color='green', marker='.', s=100,
                             label=f"{name} Predicted Acc.")
            axarr[0].scatter(df[df[model_name + "label"] == 0].index,
                             df["portfolio_value"][df[model_name + "label"] == 0], color='red', marker='.', s=100,
                             label=f"{name} Predicted Slo.")
            axarr[1].plot(100 * portfolio_history["return_pct"], label=f"{name} Portfolio returns")
            axarr[1].grid(True)

        axarr[0].plot(self.risky_assets['price'], color='black', label='B&H SP500')
        axarr[0].grid(True)

        axarr[0].legend(loc='best')
        axarr[1].legend(loc='best')

        plt.show()

