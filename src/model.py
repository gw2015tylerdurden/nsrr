import torch
import torch.nn as nn
from .utils import remove_padding_batch_data, add_padding_batch_data
from .base import ModelBase

'''
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
            data = remove_padding_data(x)

            out = conv_layer(data)
            outputs.append(out)

        padding_out = add_padding_data(outputs)
        return padding_out


class MaxPool1dDifferentSamplingFreq(nn.Module):
    def __init__(self, fs_channels, kernel_size, stride):
        super().__init__()
        self.max_pools = nn.ModuleList()

        for fs in fs_channels:
            kernel_size = int(base_kernel_size * (max_fs / fs))
            max_pool = nn.MaxPool1d(kernel_size, stride)
            self.max_pools.append(max_pool)

    def forward(self, x):
        outputs = []
        num_channels = channel_data.size(1)

        for i in range(num_channels):
            data = remove_padding_data(channel_data)

            out = self.max_pool(data)
            outputs.append(out)

        pooling_out = add_padding_data(outputs)
        return pooling_out
'''

class ModelCNN(ModelBase):
    def __init__(self, num_classes, fs_channels, feature_size=10):
        super().__init__()
        one_channel = 1
        conv_feat_num = 256
        fc_size = int(conv_feat_num * len(fs_channels) * feature_size)
        self.features = nn.ModuleList()

        for fs in fs_channels:
            self.features.append(nn.Sequential(
                nn.Conv1d(one_channel, 32, kernel_size=5, stride=1, padding=2),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=2),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=1),

                nn.Conv1d(128, conv_feat_num, kernel_size=2, stride=1, padding=1),
                nn.GELU(),

                nn.AdaptiveAvgPool1d(feature_size)
            ))

        self.classifier = nn.Sequential(
            nn.Linear(fc_size, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = []
        for i, layers in enumerate(self.features):
            channel_data = x[:, i, :, :]
            signal = remove_padding_batch_data(channel_data)
            features.append(layers(signal))

        # flatten
        x = torch.cat([f.view(f.size(0), -1) for f in features], dim=1)
        #flatten = [remove_padding_data(x[:, i, :, :]).view(batch, -1) for i in range(channel)]
        x = self.classifier(x)
        return x
