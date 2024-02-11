import torch
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .cam1d import GradCAM1D
from .model import ModelCNN
from .plots import plot_cam_1d, plot_sim


class CamCalculator():
    def __init__(self, model_files, device, annotation_labels, fs_channels, channel_labels, all_predictions):
        self.model_files = model_files
        self.device = device
        self.annotation_labels = annotation_labels
        self.fs_channels = fs_channels
        self.channel_labels = channel_labels
        self.all_predictions = all_predictions

    def calc_sim_result(self, data_loader):
        sim, label_counts = self.calc_signals_importance_matrix(data_loader)
        np.save(f'signal_importance_matrix.npy', sim)
        np.save(f'label_counts.npy', label_counts)
        plot_sim(sim, self.channel_labels, self.annotation_labels, label_counts)
        # get signal importance vector (Eq. (3.14))
        sim_vector = sim.mean(axis=1)
        return np.argmin(sim_vector), np.argmax(sim_vector)


    def __calc_sim(self, model_file, data_loader):
        num_labels = len(self.annotation_labels)
        num_channels = len(self.channel_labels)

        signals_importance_matrix = np.zeros((num_channels, num_labels))
        label_counts = np.zeros(len(self.annotation_labels))

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            targets = [ClassifierOutputTarget(i.item()) for i in self.all_predictions]
            reload_model = ModelCNN(len(self.annotation_labels), self.fs_channels).model_instance

            for channel_idx, feature in enumerate(reload_model.features):
                reload_model.load_state_dict(torch.load(model_file))
                last_layer = [feature[-1]]
                cam = GradCAM1D(model=reload_model, target_layers=last_layer)
                _ = cam(input_tensor=inputs, targets=targets)

                labels_np = labels.cpu().numpy()
                for label_idx in np.unique(labels_np):
                    label_indices = np.where(labels_np == label_idx)[0]
                    cam_alpha_for_label = cam.alpha[label_indices]
                    # get non negative values (Eq. (3.11))
                    g = np.mean(np.maximum(cam_alpha_for_label, 0), axis=1)
                    # get signal importance  (Eq. (3.10))
                    L_batch = np.sum(g) / len(label_indices)
                    signals_importance_matrix[channel_idx, label_idx] += L_batch

                    # for plot result
                    label_counts[label_idx] += len(label_indices)
        return signals_importance_matrix, label_counts


    def calc_signals_importance_matrix(self, data_loader):

        sims = []
        for model_file in self.model_files:
            sim, label_counts = self.__calc_sim(model_file, data_loader)
            sims.append(sim)

        # ensamble
        signals_importance_matrix = np.mean(sims, axis=0)

        return signals_importance_matrix, label_counts

    def plot_cam(self, data_loader):
        plot_once = True
        all_model_cams = []

        for model_file in self.model_files:
            # set cam targets as the predicred label
            targets = [ClassifierOutputTarget(i.item()) for i in self.all_predictions]
            reload_model = ModelCNN(len(self.annotation_labels), self.fs_channels).model_instance

            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                cams = []
                for feature in reload_model.features:
                    reload_model.load_state_dict(torch.load(model_file))
                    last_layer = [feature[-1]]
                    cam = GradCAM1D(model=reload_model, target_layers=last_layer)
                    result = cam(input_tensor=inputs, targets=targets)
                    cams.append(result)

                plot_cam_1d(inputs, labels, np.array(cams), self.annotation_labels, self.channel_labels, self.fs_channels, fname=model_file)
                if plot_once is True:
                    break
            all_model_cams.append(cams)

        ensemble_cam = np.mean(np.array(all_model_cams), axis=0)
        plot_cam_1d(inputs, labels, ensemble_cam, self.annotation_labels, self.channel_labels, self.fs_channels, fname="ensemble")
