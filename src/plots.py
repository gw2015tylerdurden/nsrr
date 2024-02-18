import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.metrics import confusion_matrix
from scipy.interpolate import interp1d

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
        annot=True,
        fmt='d'  # integer
        cbar=True
        normalize='true'
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

    
def plot_and_save(self, inputs, labels, cams, annotation):
    each_labels, indices = np.unique(labels.detach().cpu().numpy(), return_index=True)
    channel_num = inputs.size(1)
    target_size = inputs.size(3)

    def scale_cam_1d(cam, target_size):
        x_old = np.linspace(0, 1, len(cam))
        x_new = np.linspace(0, 1, target_size)
        interpolator = interp1d(x_old, cam, kind='linear')
        cam_resized = interpolator(x_new)
        return cam_resized

    for label_idx in indices:
        fig, axs = plt.subplots(channel_num, 1, figsize=(18, 2 * channel_num))
        input_data = inputs[label_idx].squeeze().detach().cpu().numpy()

        for i in range(channel_num):
            cam_data = scale_cam_1d(cams[i][label_idx].squeeze(), target_size)
            colors = plt.cm.jet(cam_data / np.max(cam_data))
            point_sizes = cam_data[i] * 10

            axs[i].plot(input_data[i], label='Input Signal', color='black', alpha=0.7)
            axs[i].scatter(np.arange(len(cam_data)), input_data[i], color=colors, s=point_sizes, label='CAM')
            axs[i].set_title(f'{self.channel_labels[i]}')
        axs[0].legend(loc='upper right')
        plt.title(f'Label: {annotation[labels[label_idx].item()]}')
        plt.tight_layout()
        filename = f"cam_{self.plot_epoch}_{annotation[labels[label_idx]]}.svg"
        plt.savefig(filename)
        plt.close()
    print(f"[LOG] class activation map is saved")

