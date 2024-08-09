import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')

def plot_confusion_matrix(y_true: np.array,
                          y_pred: np.array,
                          labels: list,
                          title: str='Confusion matrix',
                          cmap: str='Blues') -> plt.Figure:
    """
    Plots a confusion matrix

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param labels: List of class labels
    :param title: Title of the plot
    :param cmap: Color map
    :return: Figure
    """
    # Round predicitons if nessesary
    if y_pred.dtype == float:
        y_pred = np.round(y_pred)

    if y_true.dtype == float:
        y_true = np.round(y_true)

    cm = confusion_matrix(y_true, y_pred).astype(int)
    try:
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    except:
        df_cm = pd.DataFrame(cm)
    fig = plt.figure(figsize=(10,7))
    sns.heatmap(df_cm,
                annot=True,
                cmap=cmap,
                annot_kws={"fontsize": 16, "weight": 'bold'},
                fmt='g'
                )
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig

def plot_continous_confussion_matrix(y_true: np.array,
                          y_pred: np.array,
                          labels: list,
                          title: str='Confusion matrix',
                          cmap: str='Blues') -> plt.Figure:
    """
    Plots a confusion matrix

    """
    # Remove second dimension from y_pred
    y_pred = y_pred.squeeze()

    # Create dotplot
    fig = plt.figure(figsize=(10,7))
    sns.scatterplot(x=y_pred, y=y_true)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Draw labels to y-axis
    plt.yticks(ticks=[0,1,2], labels=labels)

    # Invert Y-axis
    plt.gca().invert_yaxis()

    # Draw lines between descision boundaries
    plt.axvline(x=1.5, color='black', linestyle='--')
    plt.axvline(x=0.5, color='black', linestyle='--')

    #print('stop')

    # save figure
    #plt.savefig('test.png')

    return fig
