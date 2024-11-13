import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.metrics import confusion_matrix
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize
from .utils import remove_padding_data

matplotlib.use('Agg')

def plot_confusion_matrix(y_true, y_pred, classes, accuracy, epoch, title=None, cmap=plt.cm.Purples):
    if not title:
        title = f'Confusion matrix (Accuracy: {accuracy:.2f}%)'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # insert REM between Wake and Stage 1
    if True:
        new_order = [0, 5, 1, 2, 3, 4]
        classes = ['Wake', 'REM', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
        cm = cm[new_order, :]
        cm = cm[:, new_order]
    cm_normalized = normalize(cm, axis=1, norm='l1')

    annot = np.vectorize(lambda x: f'{x:.1%}' if x != 0 else '')(cm_normalized)

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
        cbar=True,
        cbar_kws={'pad': 0.15}
    )

    # Adding counts of true labels on a new axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([i + 0.5 for i in range(len(classes))])
    ax2.set_yticklabels(np.sum(cm, axis=1))

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


def plot_cam_1d(inputs, labels, cams, annotation, channel_labels, fs_channels, fname=''):
    each_labels, indices = np.unique(labels.detach().cpu().numpy(), return_index=True)
    channel_num = len(channel_labels)

    def scale_cam_1d(cam, target_size):
        x_old = np.linspace(0, 1, len(cam))
        x_new = np.linspace(0, 1, target_size)
        interpolator = interp1d(x_old, cam, kind='linear')
        cam_resized = interpolator(x_new)
        return cam_resized

    for label_idx in indices:
        fig, axs = plt.subplots(channel_num, 1, figsize=(18, 2 * channel_num))
        data = inputs[label_idx].squeeze()
        all_channel_cam = cams[:, label_idx, :].squeeze()

        for i in range(channel_num):
            input_data = remove_padding_data(data[i]).detach().cpu().numpy()
            target_size = len(input_data)
            time = np.linspace(0,  target_size / fs_channels[i], target_size)
            if np.max(all_channel_cam) != 0:
                cam_data_normalized = (all_channel_cam[i] - np.min(all_channel_cam)) / (np.max(all_channel_cam) - np.min(all_channel_cam))
                cam_data = scale_cam_1d(cam_data_normalized, target_size)
            else:
                cam_data = np.zeros(target_size)
                print(f"[WARN] all class activation map is 0")

            point_sizes = 10

            axs[i].plot(time, input_data, label='Input Signal', color='black', alpha=0.2)
            scatter = axs[i].scatter(time, input_data, c=cam_data, cmap='jet', s=point_sizes)
            axs[i].set_title(f'{channel_labels[i]}')
        fig.subplots_adjust(hspace=0.5, right=0.85)
        fig.colorbar(scatter, ax=axs.ravel().tolist(), orientation='vertical', pad=0.01, aspect=20, cmap='jet')
        axs[0].legend(loc='upper right')
        plt.suptitle(f'Label: {annotation[labels[label_idx].item()]}')
        filename = f"cam_{annotation[labels[label_idx]]}_{fname}.svg"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    print(f"[LOG] class activation map is saved")


def plot_sim(sim, channel_labels, annotation, label_counts):
    sim_with_siv = np.hstack((sim, np.mean(sim, axis=1, keepdims=True)))
    annotation = annotation + ['SIV']
    min_vals = np.min(sim_with_siv, axis=0)
    max_vals = np.max(sim_with_siv, axis=0)
    sim_normalized = (sim_with_siv - min_vals) / (max_vals - min_vals)

    fig, ax = plt.subplots(figsize=(12, 8))
    seaborn.heatmap(sim_normalized, cmap='viridis', ax=ax, annot=True, fmt=".2f")

    ax.set_xticks(np.arange(len(annotation)) + 0.5)
    ax.set_yticks(np.arange(len(channel_labels)) + 0.5)
    ax.set_xticklabels(annotation, rotation=45, ha='right')
    ax.set_yticklabels(channel_labels, rotation=0)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot label_counts
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(len(annotation)) + 0.5)
    ax2.set_xticklabels(np.append(label_counts, ''), rotation=0, ha='right')

    plt.title(f'Signal Importance Matrix', pad=20)
    plt.tight_layout()
    # Save figure as SVG with epoch number
    filename = f"signals_importance_matrix.svg"
    plt.savefig(filename)
    print(f"[LOG] Signal Importance Matrix saved to {filename}")
    plt.close()
