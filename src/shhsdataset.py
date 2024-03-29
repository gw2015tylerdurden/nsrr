import torch
import copy
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import h5py
import ast
from collections import defaultdict
import numpy as np

class ShhsDataset(Dataset):
    def __init__(self, h5_files, pop_channels=None, noise_channels_fs=None, duration=30.0):
        self.h5_files = h5_files
        self.dataset_patient_keys = []
        self.h5_file_objects = []

        for h5_file in h5_files:
            file = h5py.File(h5_file, 'r')
            self.h5_file_objects.append(file)
            self.dataset_patient_keys.extend([(file, key) for key in file.keys() if "shhs" in key])

        file = self.h5_file_objects[0]
        byte_string_to_list = lambda b: ast.literal_eval(b.decode('utf-8'))
        self.annotation_labels = byte_string_to_list(file["annotation_labels"][()])
        self.target_fs = file["target_fs"][()]
        self.noise_channels_fs = noise_channels_fs
        self.duration = duration

        self.original_channels = byte_string_to_list(file["channels"][()])
        self.original_fs_channels = file["fs_channels"][:]

        if noise_channels_fs is not None and noise_channels_fs != []:
            self.original_channels = np.append(self.original_channels, ["Noise " + str(x) + "Hz" for x in noise_channels_fs])
            self.original_fs_channels = np.append(self.original_fs_channels, noise_channels_fs)

        self.pop_channels = pop_channels
        if pop_channels is not None:
            self.channels, self.fs_channels = self.__get_target_channels()
        else:
            self.channels = self.original_channels
            self.fs_channels = self.original_fs_channels


    def __get_target_channels(self):
        channels = []
        fs_channels = []
        for i, channel in enumerate(self.original_channels):
            if channel not in self.pop_channels:
                channels.append(self.original_channels[i])
                fs_channels.append(self.original_fs_channels[i])
            else:
                # pop
                continue

        return channels, fs_channels

    def __len__(self):
        return len(self.dataset_patient_keys)


    def __getitem__(self, idx):
        file, patient, event = self.dataset_patient_keys[idx]
        original_signals = file[patient][event]['signal'][:]
        label = file[patient][event]['label'][()]
        signals = self.__selelt_signals(original_signals)

        original_data = [torch.tensor(s, dtype=torch.float32) for s in signals]
        padding_data = pad_sequence(original_data, batch_first=True, padding_value=float('nan')).unsqueeze(1)
        label = torch.tensor(label)
        return padding_data, label

    def __selelt_signals(self, original_signals):
        signals = []
        for i, channel_label in enumerate(self.original_channels):
            if self.pop_channels is not None and channel_label in self.pop_channels:
                # pop channel
                continue
            else:
                # Replace specified channels with noise
                if "Noise" in channel_label:
                    signal = np.random.normal(0, 1, int(self.original_fs_channels[i] * self.duration))
                else:
                    signal = original_signals[i]

                signals.append(signal)
        return signals


    def get_dataset_info(self):
        return self.fs_channels, self.channels, self.target_fs, self.annotation_labels


    def __is_satisfied_datanum(self, balance_data_num, label_event_count):
        # This method needs to check if each label has at least `balance_annotation_num` events
        # Assuming `label_event_count` is a dictionary that tracks the number of events for each label
        for label, count in label_event_count.items():
            if count < balance_data_num:
                return False
        return True

    def __collect_balanced_data(self, balance_data_num, used_patients=set()):
        balanced_data_map = defaultdict(list)
        label_event_count = defaultdict(int, {label: 0 for label in range(len(self.annotation_labels))})

        for file, patient in self.dataset_patient_keys:
            if patient in used_patients:
                # for test
                continue
            for event in file[patient].keys():
                label = file[patient][event]['label'][()]
                if label_event_count[label] >= balance_data_num:
                    continue
                else:
                    balanced_data_map[(file, patient)].append(event)
                    label_event_count[label] += 1
                    used_patients.add(patient)
            if self.__is_satisfied_datanum(balance_data_num, label_event_count):
                break
        return balanced_data_map, used_patients

    def balance_dataset(self, train_data_num=2000, test_data_num=1000):
        train_dataset = copy.copy(self)
        test_dataset = copy.copy(self)
        train_keys = []
        test_keys = []

        np.random.shuffle(self.dataset_patient_keys)
        train_balanced_data_map, used_patients = self.__collect_balanced_data(train_data_num, set())
        for (file, patient), events in train_balanced_data_map.items():
            for event in events:
                train_keys.append((file, patient, event))
        np.random.shuffle(train_keys)

        test_balanced_data_map, _ = self.__collect_balanced_data(test_data_num, used_patients)
        for (file, patient), events in test_balanced_data_map.items():
            for event in events:
                test_keys.append((file, patient, event))
        np.random.shuffle(test_keys)

        train_dataset.dataset_patient_keys = train_keys
        test_dataset.dataset_patient_keys = test_keys

        return train_dataset, test_dataset


    def split(self, size=0.7, random_state=None):

        labels = [file[f"{key}/label"][()] for file, key in self.dataset_patient_keys]

        train_idx, test_idx = train_test_split(
            list(range(self.__len__())),
            train_size=size,
            random_state=random_state,
            stratify=labels
        )

        train_dataset = copy.copy(self)
        train_dataset.dataset_info = [self.dataset_patient_keys[i] for i in train_idx]

        test_dataset = copy.copy(self)
        test_dataset.dataset_info = [self.dataset_patient_keys[i] for i in test_idx]

    def close(self):
        for file in self.h5_file_objects:
            file.close()
