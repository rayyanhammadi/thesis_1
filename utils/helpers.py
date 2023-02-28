import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#Performance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,\
    roc_auc_score, roc_curve, confusion_matrix ,mean_squared_error, precision_recall_curve

def compute_sharpe_ratio(monthly_return_pct):
    # Calculate daily returns

    # Calculate annualized average daily return
    avg_monthly_return = monthly_return_pct.mean()
    avg_annual_return = avg_monthly_return * 12  # 252 trading days in a year

    # Calculate annualized standard deviation of daily returns
    std_monthly_return = monthly_return_pct.std()
    std_annual_return = std_monthly_return * np.sqrt(12)

    # Calculate Sharpe Ratio
    sharpe_ratio = (avg_annual_return - 0.02) / std_annual_return  # Assume risk-free rate of 2%

    return sharpe_ratio
def plot_predictions(y_label=None,y_probs=None,name=None):
    """
    Trace les prédictions du modèle
    :return:
    """


    fig, ax = plt.subplots()
    ax.plot(y_probs.index, y_probs, color='black')
    threshold = 0.5 #Seuil par défaut
    ax.axhline(threshold, color='gray', lw=2, alpha=0.7)
    ax.fill_between(y_label.index, 0, 1, where=y_label == 1,
                    color='gray', alpha=0.5, transform=ax.get_xaxis_transform())
    plt.title(str(name))
    plt.show()
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,labels=None,preds=None):
    '''
    This function will make a pretty plot of a sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.plt.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

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
        sensitivity = cf[0,0] / sum(cf[0, :])

        f1_score_ = f1_score(y_true=labels,y_pred=preds)
        mse = mean_squared_error(labels,preds)
        roc = roc_auc_score(labels,preds)
        stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nspecificity={:0.3f}\nsensitivity={:0.3f}\nF1 Score={:0.3f}\nMse={:0.3f}" \
                     "\n ROC_AUC Score={:0.3f}".format(

            accuracy, precision, specificity,sensitivity, f1_score_,mse,roc)
    #     else:
    #         stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    # else:
    #     stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    plt.show()

def show_confusion_matrix(labels=None, preds=None,name=None):
    """

    Renvoie la matrice de confusion du modéle

    """

    confusion_df = confusion_matrix(labels,preds)
    label=["True Neg","False Pos","False Neg","True Pos"]
    categories = ["Slowdown", "Acceleration"]
    make_confusion_matrix(cf = confusion_df,group_names=label,categories=categories,cmap='binary',labels=labels
                               ,preds=preds, title=name)



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
    filename = f"{name}_predictions.txt"
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
