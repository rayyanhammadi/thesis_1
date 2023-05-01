# Project Title

Investing through Economic Cycles with Machine Learning Algorithms
## Description

This project involves the implementation and backtesting of three machine learning algorithms (logit, random forest, and xgradient boosting) using a design matrix and true labels of a binary target variable. The code is implemented in Python and uses libraries such as numpy, pandas, scikit learn, seaborn, statsmodel, shap, mlxtend.evaluate, and yahoofinancials. The project also includes enhancements such as normalization, resampling, and threshold tuning.

## Usage

There is no installation required for this project. To run the models, open the file called `run_models`.
To show plots of predictions and confusion matrices, open `visualize_predictions`.
To see backtesting results, open `backtesting_portfolio`.

## Structure

The project has the following structure:

- `Data`: This folder contains the data required to run machine learning algorithms. It includes a `.xlsx` file containing the design matrix (432x70) and true labels of the target variable (binary).
- `Main`: This folder contains the following files:
  - `backtesting.py`: This file implements portfolios and functions to assess them.
  - `data_processing.py`: This file processes data and includes normalization functions, etc.
  - `modelization.py`: This file implements machine learning algorithms and methods to improve models.
- `Results`: This folder contains empirical results, confusion matrices, plots of predictions, etc.
- `Test`: This folder contains the following files:
  - `run_models.py`: This file runs the machine learning models.
  - `portfolio_backtesting`: This file performs portfolio backtesting.
  - `visualize_predictions`: This file shows plots of predictions.
- `Utils`: This folder contains the `helpers.py` functions that are useful.


## Contributing

New machine learning and neural network models, new methods to improve models, and new variables to the design matrix can be added to this project.

## Issues

Threshold tuning for `meth_1` in `modelization.py` is not implemented, but everything else executes correctly.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

