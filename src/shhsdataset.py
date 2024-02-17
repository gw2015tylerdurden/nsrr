import torch
import copy
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import h5py

class ShhsDataset(Dataset):
    def __init__(self, h5_files):
        self.h5_files = h5_files
        self.dataset_keys = []
        self.h5_file_objects = []

        for h5_file in h5_files:
            file = h5py.File(h5_file, 'r')
            self.h5_file_objects.append(file)
            self.dataset_keys.extend([(file, key) for key in file.keys() if "shhs" in key])


    def __len__(self):
        return len(self.dataset_keys)


    def __getitem__(self, idx):
        file, dataset_name = self.dataset_keys[idx]

        # Convert from float64 to float32
        original_data = [torch.tensor(s, dtype=torch.float32) for s in file[f"{dataset_name}/signal"][:]]
        padding_data = pad_sequence(original_data, batch_first=True, padding_value=float('nan')).unsqueeze(1)
        label = torch.tensor(file[f"{dataset_name}/label"][()])
        return padding_data, label


    def get_dataset_info(self):
        file = self.h5_file_objects[0]
        byte_string_to_list = lambda b: ast.literal_eval(b.decode('utf-8'))
        channels = byte_string_to_list(file["channels"][()])
        annotation_labels = byte_string_to_list(file["annotation_labels"][()])
        return file["fs_channels"][:], channels, file["target_fs"][()], annotation_labels


    def split(self, size=0.7, random_state=None):

        labels = [file[f"{key}/label"][()] for file, key in self.dataset_keys]

        train_idx, test_idx = train_test_split(
            list(range(self.__len__())),
            train_size=size,
            random_state=random_state,
            stratify=labels
        )

        train_dataset = copy.copy(self)
        train_dataset.dataset_info = [self.dataset_keys[i] for i in train_idx]

        test_dataset = copy.copy(self)
        test_dataset.dataset_info = [self.dataset_keys[i] for i in test_idx]

        return train_dataset, test_dataset

    def close(self):
        for file in self.h5_file_objects:
            file.close()
