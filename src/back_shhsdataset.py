import torch
import copy
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split


class ShhsDataset(Dataset):
    def __init__(self, dataset_info, channel_labels, fs=128.0, duration=30.0, interp='cubic', norm='standard'):
        self.dataset_info = dataset_info
        self.target_fs = fs
        self.duration = duration


    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, idx):
        info = self.dataset_info[idx]
        fs_channels, extracted_data = self.__extract_data(info)
        interpolated_data = self.__interpolate_data(fs_channels, extracted_data)
        normalized_data = [self.normalizer.normalize(data) for data in interpolated_data]

        data = torch.tensor(np.array(normalized_data), dtype=torch.float32, requires_grad=False)
        return data, info['label_idx']

    def split(self, train_size=0.7, random_state=None):

        labels = [info['label_idx'] for info in self.dataset_info]

        train_idx, test_idx = train_test_split(
            list(range(len(self.dataset_info))),
            train_size=train_size,
            random_state=random_state,
            stratify=labels
        )

        train_dataset = copy.copy(self)
        train_dataset.dataset_info = [self.dataset_info[i] for i in train_idx]

        test_dataset = copy.copy(self)
        test_dataset.dataset_info = [self.dataset_info[i] for i in test_idx]

        return train_dataset, test_dataset
