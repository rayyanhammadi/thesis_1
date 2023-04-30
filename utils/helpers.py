import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

import seaborn as sns
from mlxtend.evaluate import mcnemar_table, mcnemar

#Performance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,\
    roc_auc_score, roc_curve, confusion_matrix ,mean_squared_error, precision_recall_curve, log_loss


def plot_predictions(y_label=None, y_probs: list = None, opt_threshold: list = None, names=None, acc: list = None):
    """
    Plots predicted probabilities or binary classifications of multiple models over time.

    Args:
    y_label (pd.Series): Actual binary classification labels for the target variable
    y_probs (list of pd.Series): Predicted probabilities or binary classifications for each model
    opt_threshold (list of float or None): "Optimal" classification threshold for each model.
        If None, default threshold of 0.5 is used.
    names (list of str): Name of each model to use as label for the plot
    acc (list of pd.Series) : Accuracy of the model imputing unknown Y's
        If not None plot the accuracy for each model

    Returns:
    None
    """

    # Create a plot with the specified size
    fig, ax = plt.subplots(figsize=(12, 8))

    # For each set of predicted probabilities or binary classifications
    for i, y_prob in enumerate(y_probs):
        # Plot the probabilities or classifications over time and add a label with the model name
        ax.plot(y_prob.index, y_prob, label=names[i])

        # If "optimal" classification threshold is not provided, set it to default threshold of 0.5
        if opt_threshold is None:
            threshold = 0.5
            # Draw a horizontal line at the default threshold for comparison
            ax.axhline(threshold, color='grey', lw=2, alpha=0.7)
        else:
            # If "optimal" classification threshold is provided, plot it as a line
            ax.plot(opt_threshold[i], color='grey', lw=2, alpha=0.7, label='"Optimal" Threshold')

    if acc is not None:
        for i, acc in enumerate(acc):
            ax.plot(acc.iloc[204:], label="Accuracy of predicting the last 11 missing Y's", linestyle="-")
            # Add point to accuracy plot


    # Shade areas on the plot where the actual labels are 1 or 0
    ax.fill_between(y_label.index, 0, 1, where=y_label == 1,
                    color='green', alpha=0.1, transform=ax.get_xaxis_transform())
    ax.fill_between(y_label.index, 0, 1, where=y_label == 0,
                    color='red', alpha=0.1, transform=ax.get_xaxis_transform())

    # Add a title and labels for the x and y axes
    ax.set_title('Predictions')
    ax.set_xlabel('Time')
    ax.legend(loc='best')
    plt.show()


def compute_sharpe_ratio(monthly_return_pct):
    """
    Compute Sharpe Ratio given monthly returns in percentage

    Args:
    - monthly_return_pct (pandas.Series): Monthly returns in percentage

    Returns:
    - sharpe_ratio (float): The Sharpe Ratio of the given monthly returns
    """

    # Calculate annualized average daily return
    avg_monthly_return = monthly_return_pct.mean()
    avg_annual_return = avg_monthly_return * 12  # 252 trading days in a year

    # Calculate annualized standard deviation of daily returns
    std_monthly_return = monthly_return_pct.std()
    std_annual_return = std_monthly_return * np.sqrt(12)

    # Calculate Sharpe Ratio : Assume risk-free rate of 2%
    sharpe_ratio = (avg_annual_return - 0.02) / std_annual_return

    return sharpe_ratio



def make_confusion_matrices(cfs: list,
                            group_names=None,
                            categories: list = None,
                            count=True,
                            percent=True,
                            cbar=True,
                            xyticks=True,
                            xyplotlabels=True,
                            sum_stats=True,
                            figsize=None,
                            cmap='Blues',
                            title=None,
                            labels= None,
                            preds: list = None):

    blanks = ['' for i in range(cfs[0].size)]

    if group_names and len(group_names) == cfs[0].size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    fig, axs = plt.subplots(1, len(cfs), figsize=figsize)

    for i, cf in enumerate(cfs):
        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:

            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace(cf) / float(np.sum(cf))

            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            specificity = cf[1, 1] / sum(cf[1, :])
            sensitivity = cf[0, 0] / sum(cf[0, :])

            f1_score_ = f1_score(y_true=labels, y_pred=preds[i])
            mse = mean_squared_error(labels, preds[i])
            roc = roc_auc_score(labels, preds[i])
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nspecificity={:0.3f}\nsensitivity={:0.3f}\nF1 Score={:0.3f}\nMse={:0.3f}" \
                         "\n ROC_AUC Score={:0.3f}".format(

                accuracy, precision, specificity, sensitivity, f1_score_, mse, roc)

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if xyticks == False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories,
                    ax=axs[i])

        if xyplotlabels:
            axs[i].set_ylabel('True label')
            axs[i].set_xlabel('Predicted label' + stats_text)
        else:
            axs[i].set_xlabel(stats_text)

        if title:
            axs[i].set_title(title[i])

    plt.show()








def show_confusion_matrix(labels ,  preds:list ,names: list):
    """

    Renvoie la matrice de confusion du modéle

    """
    label=["True Neg","False Pos","False Neg","True Pos"]
    categories = ["Slowdown", "Acceleration"]

    confusion_dfs = []
    for i, pred in enumerate(preds):
        confusion_dfs.append(confusion_matrix(labels,pred))

    make_confusion_matrices(cfs = confusion_dfs,categories=categories,group_names=label,labels=labels
                               ,preds=preds, title=names)



def store_predictions(y_hat, name):
    """
    Stores the predicted labels and probabilities for a given method in a .txt file.

    Parameters:
    -----------
    y_hat_label : pandas DataFrame
        The predicted labels.
    y_hat_probs : pandas DataFrame
        The predicted probabilities.
    name : str
        The name of the method used to make the predictions.

    Returns:
    --------
    None
    """
    # create a new pandas dataframe to hold the predictions and probabilities

    # write the data to a text file
    filename = f"{name}.txt"
    y_hat.to_csv(filename, sep='\t', index=True)



def read_predictions(filename):
    """
    Reads predicted labels and probabilities from a .txt file.

    Parameters:
    -----------
    filename : str
        The name of the .txt file containing the predictions.

    Returns:
    --------
    y_hat_label : pandas DataFrame
        The predicted labels.
    y_hat_probs : pandas DataFrame
        The predicted probabilities.
    name : str
        The name of the method used to make the predictions.
    """
    # read the data from the file into a pandas dataframe
    y_hat = pd.read_csv(filename, sep='\t')
    y_hat["dates"] = pd.to_datetime(y_hat["dates"])
    y_hat.set_index("dates", drop=True, inplace=True)
    y_hat = y_hat.astype("float")
    # separate the labels and probabilities into separate dataframes

    # extract the method name from the filename
    #name = filename.split('_')[0]

    return y_hat


def compare_models(y_true, y_model1, y_model2,alpha=.05):
    """
    Compare the performance of two models using McNemar's test.

    Args:
        y_true (numpy.ndarray): True binary labels for each observation in the test data.
        y_model1 (numpy.ndarray): Predicted binary outcomes for each observation in the test data using model 1.
        y_model2 (numpy.ndarray): Predicted binary outcomes for each observation in the test data using model 2.

    Returns:
        tuple: A tuple containing the test result (reject or fail to reject the null hypothesis),
               the p-value of the test, and the test statistic.
    """

    H0 = "There is no significant difference in the performance of the two models"
    H1 = "One of the models performs significantly better than the other"

    # Calculate the counts of true positives, false positives, false negatives, and true negatives for each model.
    tb = mcnemar_table(y_target=y_true,
                       y_model1=y_model1,
                       y_model2=y_model2)

    # Calculate the McNemar's test statistic and p-value.
    chi2, p_value = mcnemar(ary=tb, corrected=True)


    # Determine whether to reject or fail to reject the null hypothesis based on the p-value.
    if p_value < 0.05:
        test_result = "Reject the null hypothesis"
    else:
        test_result = "Fail to reject the null hypothesis"

    # Determine the best performing model in terms of the decision variable.
    model1_wins = 0
    model2_wins = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_model1[i] == 1 and y_model2[i] == 0:
            model1_wins += 1
        elif y_true[i] == 1 and y_model1[i] == 0 and y_model2[i] == 1:
            model2_wins += 1
        elif y_true[i] == 0 and y_model1[i] == 1 and y_model2[i] == 0:
            model2_wins += 1
        elif y_true[i] == 0 and y_model1[i] == 0 and y_model2[i] == 1:
            model1_wins += 1

    if model1_wins > model2_wins:
        best_model = "Model 1"
    elif model2_wins > model1_wins:
        best_model = "Model 2"
    else:
        best_model = "Both models perform equally well"
    results = f"Results of McNemar's test:\nNull Hypothesis (H0): {H0}\n" \
              f"Alternative Hypothesis (H1): {H1}\n\nMcNemar Table:\n{tb}\n" \
              f"\nMcNemar's test statistic: {chi2:.3f}\np-value: {p_value:.3f}\nSignificance level: {alpha:.3f}\n\n{test_result} " \
              f"\n\nBest performing model in terms of the decision variable: {best_model}"

    print(results)


    return test_result, p_value, chi2


def get_optimal_thresholds(models, X, Y):
    """
    This function takes in a dictionary of models, a dataset of features, and a dataset of true labels.
    It returns a dictionary of optimal thresholds for each model in the given dictionary.

    Args:
    - models: a dictionary of trained models
    - X: a dataset of features associated with the dataset of true labels
    - Y: a dataset of  true labels

    Returns:
    - opt_thresholds: a dictionary of optimal thresholds for each model in the given dictionary
    """
    opt_thresholds = {}
    for name, model in models.items():
        # Get ROC and Precision-Recall curves for the current model
        fpr, tpr, thresholds = roc_curve(Y, model.predict_proba(X)[:, 1])
        precision, recall, thresholds = precision_recall_curve(Y, model.predict_proba(X)[:, 1])

        # Calculate Metrics for the current model
        gmeans = np.sqrt(tpr * (1 - fpr))
        J = tpr - fpr
        fscore = (2 * precision * recall) / (precision + recall)

        # Evaluate thresholds for the current model
        scores = [log_loss(Y, (model.predict_proba(X)[:, 1] >= t).astype('int')) for t in
                  thresholds]
        ix_1 = np.argmin(scores)
        ix_2 = np.argmin(np.abs(J))
        ix_3 = np.argmax(fscore)
        ix_4 = np.argmax(gmeans)
        opt_threshold = {"log_score": thresholds[ix_1], "g_means": thresholds[ix_4], "J": thresholds[ix_2],
                         "f1_score": thresholds[ix_3]}

        opt_thresholds[name] = opt_threshold

    return opt_thresholds


def plot_before_after_transformation(data, transformed_data):

    x = np.arange(len(data))
    col = data.columns

    # Get the mean and standard deviation of the data before and after transformation
    stats_before, stats_after = data.describe(), transformed_data.describe()
    stats_before, stats_after = stats_before.loc[["mean", "std"]].T, stats_after.loc[["mean", "std"]].T

    # Create a figure with 3 rows and 2 columns of subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    fig_, axs_ = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # Plot the data before transformation on the first subplot
    axs[0].plot(x, data, label=col)
    axs[0].set_title('Before Transformation')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Data Values')

    # Plot the data after transformation on the second subplot
    axs[1].plot(x, transformed_data, label=col)
    axs[1].set_title('After Transformation')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Transformed Data Values')



    # Plot the mean and standard deviation of the data before transformation on the third subplot
    axs_[0].bar(x=stats_before.index, height=stats_before["mean"], yerr=stats_before["std"], width=0.3, align='center')
    axs_[0].set_title('Before Transformation')
    axs_[0].set_ylabel('Meand and Std')


    # Plot the mean and standard deviation of the data after transformation on the fourth subplot
    axs_[1].bar(x=stats_after.index, height=stats_after["mean"], yerr=stats_after["std"], width=0.3, align='center')
    axs_[1].set_title('After Transformation')
    axs_[1].set_ylabel('Mean and Std')




    # Add legend and grid to all subplots
    for ax in axs.flat:
        ax.legend(loc='best')
        ax.grid(True)

    for ax in axs_.flat:
        ax.legend(loc='best')
        ax.grid(True)

    plt.show()



def plot_autocorrelation(df):
    """
    Plots the autocorrelation of a given time series.

    Args:
    df (pd.DataFrame): a pandas DataFrame containing the time series data.

    Returns:
    None: The function displays the plot.
    """

    # Create a figure and axis object for the plot
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot the autocorrelation function with lags up to 60
    plot_acf(df, ax=ax, lags=60)

    # Set the x and y labels and the title for the plot
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')




def plot_feature_importance(model, X):
    # Calculate SHAP values
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

    # Calculate feature importance
    feature_importance = np.abs(shap_values.values).mean(0)
    feature_names = X.columns.tolist()

    # Sort feature importance in descending order
    sorted_idx = np.argsort(feature_importance)[::-1]

    # Plot feature importance with SHAP values

    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature Importance with SHAP Values')
    plt.show()

def plot_shap(models, X):
    explainer_rf = shap.TreeExplainer(models["rf_"])
    explainer_gb = shap.TreeExplainer(models["gb_"])
    explainer_logit = shap.explainers.Linear(models["logit_"],X)

    shap_values_rf = explainer_rf.shap_values(X)
    shap_values_gb = explainer_gb.shap_values(X)
    shap_values_logit = explainer_logit.shap_values(X)


    shap.summary_plot(shap_values_rf, features=X, feature_names=X.columns)
    shap.summary_plot(shap_values_gb, features=X, feature_names=X.columns)
    shap.summary_plot(shap_values_logit, features=X, feature_names=X.columns)


