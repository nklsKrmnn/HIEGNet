import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion matrix', cmap='Blues'):
    """
    Plot confusion matrix
    """

    cm = confusion_matrix(y_true, y_pred).astype(int)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig = plt.figure(figsize=(10,7))
    sns.heatmap(df_cm,
                annot=True,
                cmap=cmap,
                annot_kws={"fontsize": 16, "weight": 'bold'}
                )
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig