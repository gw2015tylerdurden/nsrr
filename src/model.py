import torch
import torch.nn as nn
from tqdm import tqdm
from .base import ModelBase, TrainingRoutineBase
from torch.nn.utils.rnn import pad_sequence
import os


def remove_padding_data(x):
    batch, feat, sequence = x.shape
    valid_data_mask = ~torch.isnan(x)
    not_nan_data = torch.masked_select(x, valid_data_mask)
    return not_nan_data.reshape(batch, feat, -1)


def add_padding_data(outputs):
    batch_size = outputs[0].size(0)
    padded_outputs = []
    for batch in range(batch_size):
        batch_outputs = [o[batch].transpose(0, 1) for o in outputs]
        # pad_sequence はシーケンス長が最初の次元である必要がある
        padded_batch = pad_sequence(batch_outputs, batch_first=True, padding_value=float('nan'))
        # pad_sequenceの結果を再び転置し, チャンネル数を最初の次元に戻す
        padded_batch = padded_batch.transpose(1, 2)
        padded_outputs.append(padded_batch)

    padding_out = torch.stack(padded_outputs)
    return padding_out


class Conv1dDifferentSamplingFreq(nn.Module):
    def __init__(self, in_channels, out_channels, fs_channels, base_kernel_size=5, stride='half'):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        max_fs = max(fs_channels)
        # return 2 when half, 1 when same, else value
        self.set_stride = lambda s: 2 if s == 'half' else (1 if s == 'same' else s)

        for fs in fs_channels:
            kernel_size = int(base_kernel_size * (max_fs / fs))
            padding = kernel_size // 2
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride=self.set_stride(stride), padding=padding)
            self.conv_layers.append(conv_layer)


    def forward(self, x):
        outputs = []
        for i, conv_layer in enumerate(self.conv_layers):
            channel_data = x[:, i, :, :]
            data = remove_padding_data(channel_data)

            out = conv_layer(data)
            outputs.append(out)

        padding_out = add_padding_data(outputs)
        return padding_out


class MaxPool1dDifferentSamplingFreq(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.max_pool = nn.MaxPool1d(kernel_size, stride)

    def forward(self, x):
        outputs = []
        num_channels = x.size(1)

        for i in range(num_channels):
            channel_data = x[:, i, :, :]
            data = remove_padding_data(channel_data)

            out = self.max_pool(data)
            outputs.append(out)

        pooling_out = add_padding_data(outputs)
        return pooling_out


class SimpleCNN(ModelBase):
    def __init__(self, num_classes, fs_channels, input_shape):
        super().__init__()
        one_channel = 1

        self.features = nn.Sequential(
            Conv1dDifferentSamplingFreq(one_channel, 32, fs_channels, base_kernel_size=5, stride='same'),
            nn.GELU(),
            MaxPool1dDifferentSamplingFreq(kernel_size=2, stride=2),

            Conv1dDifferentSamplingFreq(32, 64, fs_channels, base_kernel_size=5, stride='same'),
            nn.GELU(),
            MaxPool1dDifferentSamplingFreq(kernel_size=2, stride=2),
        )

        #feature_size = self.__calc_feature_size(input_shape)

        self.classifier = nn.Sequential(
            #nn.Linear(feature_size, 256),
            nn.Linear(14912, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        batch, channel, feature, sequence = x.shape
        flatten = [remove_padding_data(x[:, i, :, :]).view(batch, -1) for i in range(channel)]
        x = torch.cat(flatten, dim=1)
        x = self.classifier(x)
        return x

    def __calc_feature_size(self, input_shape):
        # [TODO] NaNでデータ数が減少することを考慮しないといけない、本物のデータをいれるといい
        dummy_input = torch.randn(input_shape)
        with torch.no_grad():
            features_output = self.features(dummy_input)
        return features_output.view(features_output.size(0), -1).size(1)

class ModelTrainingRoutine(TrainingRoutineBase):
    def __init__(self, model, criterion='cross_entropy', optimizer='Adam', lr=1e-3):
        super().__init__(model, criterion, optimizer, lr)

    def run(self, dataset, annotation_labels, channel_labels, num_epoch, batch_size, train_size=0.7):
        train_dataset, test_dataset = dataset.split(size=train_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=os.cpu_count(), # main cpu threads_num
            pin_memory=True,
            drop_last=True,
        )

        self.model = self.model.to(self.device)

        for epoch in range(num_epoch):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epoch}", dynamic_ncols=True)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            epoch_loss = running_loss / len(train_loader.dataset)

            if self.wandb is not None:
                self.wandb.update(loss=epoch_loss)
            else:
                print(f"Epoch {epoch + 1}/{self.num_epochs} - Loss: {epoch_loss:.4f}")

