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