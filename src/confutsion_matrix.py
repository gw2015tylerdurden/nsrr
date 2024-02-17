import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.metrics import confusion_matrix
matplotlib.use('TkAgg')

def plot_confusion_matrix(y_true, y_pred, classes, accuracy, epoch, title=None, cmap=plt.cm.Blues):
    if not title:
        title = f'Confusion matrix (Accuracy: {accuracy:.2f}%)'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    seaborn.heatmap(
        data=cm,
        ax=ax,
        linewidths=0.1,
        cmap=cmap,
        yticklabels=classes,
        xticklabels=classes,
    )
    
    ax.set(title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    # Save figure as SVG with epoch number
    filename = f"confusion_matrix_epoch_{epoch}.svg"
    plt.savefig(filename)
    print(f"[LOG] Confusion Matrix saved to {filename}")
    plt.close()
