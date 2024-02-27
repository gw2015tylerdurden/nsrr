import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.metrics import confusion_matrix
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize

matplotlib.use('Agg')
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'


def plot_confusion_matrix(y_true, y_pred, classes, accuracy, epoch, title=None, cmap=plt.cm.Blues):
    if not title:
        title = f'Confusion matrix (Accuracy: {accuracy:.2f}%)'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = normalize(cm, axis=1, norm='l1')

    annot = np.vectorize(lambda x: f'{int(x)}' if x != 0 else '')(cm)

    fig, ax = plt.subplots()
    seaborn.heatmap(
        data=cm_normalized,
        ax=ax,
        linewidths=0.1,
        cmap=cmap,
        yticklabels=classes,
        xticklabels=classes,
        annot=annot,
        fmt='',
        cbar=True
    )

    ax.set(title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    # Save figure as SVG with epoch number
    filename = f"confusion_matrix_epoch{epoch}.svg"
    plt.savefig(filename)
    print(f"[LOG] Confusion Matrix saved to {filename}")
    plt.close()


def plot_cam_1d(inputs, labels, cams, annotation, channel_labels, fs_channels, epoch):
    each_labels, indices = np.unique(labels.detach().cpu().numpy(), return_index=True)
    channel_num = inputs.size(1)

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
            target_size = len(input_data[i])
            time = np.linspace(0,  target_size / fs_channels[i], target_size)
            all_channel_cam = cams[i].squeeze()
            cam_data_normalized = (cams[i][label_idx].squeeze() - np.min(all_channel_cam)) / (np.max(all_channel_cam) - np.min(all_channel_cam))
            cam_data = scale_cam_1d(cam_data_normalized, target_size)
            point_sizes = 10

            axs[i].plot(time, input_data[i], label='Input Signal', color='black', alpha=0.2)
            scatter = axs[i].scatter(time, input_data[i], c=cam_data, cmap='jet', s=point_sizes)
            axs[i].set_title(f'{channel_labels[i]}')
        fig.subplots_adjust(hspace=0.5, right=0.85)
        fig.colorbar(scatter, ax=axs.ravel().tolist(), orientation='vertical', pad=0.01, aspect=20, cmap='jet')
        axs[0].legend(loc='upper right')
        plt.suptitle(f'Label: {annotation[labels[label_idx].item()]}')
        filename = f"cam_epoch{epoch}_{annotation[labels[label_idx]]}.svg"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    print(f"[LOG] class activation map is saved")


def plot_sim(sim, channel_labels, annotation, epoch):
    sim_with_siv = np.hstack((sim, np.mean(sim, axis=1, keepdims=True)))
    annotation = annotation + ['SIV']
    #sim_normalized = (sim_with_svg - np.min(sim_with_svg)) / (np.max(sim_with_svg) - np.min(sim_with_svg))
    sim_normalized = sim_with_siv

    fig, ax = plt.subplots(figsize=(12, 8))
    seaborn.heatmap(sim_normalized, cmap='viridis', ax=ax, annot=True, fmt=".2f")

    ax.set_xticks(np.arange(len(annotation)) + 0.5)
    ax.set_yticks(np.arange(len(channel_labels)) + 0.5)
    ax.set_xticklabels(annotation, rotation=45, ha='right')
    ax.set_yticklabels(channel_labels, rotation=0)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.title(f'Signal Importance Matrix', pad=20)
    plt.tight_layout()
    plt.show()

    # Save figure as SVG with epoch number
    filename = f"signals_importance_matrix_epoch{epoch}.svg"
    plt.savefig(filename)
    print(f"[LOG] Signal Importance Matrix saved to {filename}")
    plt.close()
