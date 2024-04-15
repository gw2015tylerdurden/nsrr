import torch
import torch.nn as nn
from .utils import remove_padding_batch_data, add_padding_batch_data
from .base import ModelBase
import math

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

class ModelCNN():
    def __init__(self, num_classes, fs_channels, model='default'):
        if model != 'default':
            self.model_instance = ResNet54(num_classes, fs_channels)
        else:
            self.model_instance = SimpleCNN(num_classes, fs_channels)


class SimpleCNN(ModelBase):
    def __init__(self, num_classes, fs_channels, feature_size=10):
        super().__init__()
        one_channel = 1
        conv_feat_num = 512
        fc_size = int(conv_feat_num * len(fs_channels) * feature_size)
        self.features = nn.ModuleList()

        for fs in fs_channels:
            self.features.append(nn.Sequential(
                nn.Conv1d(one_channel, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(16),
                nn.PReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.PReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(64),
                nn.PReLU(),

                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.PReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.PReLU(),

                nn.Conv1d(256, conv_feat_num, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(conv_feat_num),
                nn.PReLU(),
                nn.AdaptiveAvgPool1d(feature_size),
            ))

        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(fc_size, fc_size//100),
            nn.PReLU(),
            nn.BatchNorm1d(fc_size//100),
            nn.Dropout(0.6),
            nn.Linear(fc_size//100, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
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

# class SimpleCNN(ModelBase):
#     def __init__(self, num_classes, fs_channels, feature_size=10):
#         super().__init__()
#         one_channel = 1
#         conv_feat_num = 256
#         fc_size = int(conv_feat_num * len(fs_channels) * feature_size)
#         self.features = nn.ModuleList()
#         self.lstms = nn.ModuleList()

#         for fs in fs_channels:
#             self.features.append(nn.Sequential(
#                 nn.Conv1d(one_channel, 64, kernel_size=5, stride=1, padding=1),
#                 nn.BatchNorm1d(64),
#                 nn.ReLU(),
#                 nn.MaxPool1d(kernel_size=2, stride=2),

#                 nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=1),
#                 nn.BatchNorm1d(128),
#                 nn.ReLU(),
#                 nn.MaxPool1d(kernel_size=2, stride=2),

#                 nn.Conv1d(128, conv_feat_num, kernel_size=5, stride=1, padding=1),
#                 nn.BatchNorm1d(conv_feat_num),
#                 nn.ReLU(),
#                 nn.AdaptiveAvgPool1d(feature_size),
#             ))

#             self.lstms.append(
#                 nn.LSTM(feature_size, hidden_size=feature_size * 2, num_layers=conv_feat_num // 2, dropout=0.5, batch_first=True)
#                 )

#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(66560, 1024),
#             nn.ReLU(),
#             nn.BatchNorm1d(1024),
#             nn.Dropout(0.4),
#             nn.Linear(1024, num_classes)
#         )

#     def forward(self, x):
#         features = []
#         for i, layers in enumerate(self.features):
#             channel_data = x[:, i, :, :]
#             signal = remove_padding_batch_data(channel_data)
#             out, _ = self.lstms[i](layers(signal))
#             features.append(out)

#         # flatten
#         x = torch.cat([f.reshape(f.size(0), -1) for f in features], dim=1)
#         #flatten = [remove_padding_data(x[:, i, :, :]).view(batch, -1) for i in range(channel)]
#         x = self.classifier(x)
#         return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print('out', out.size(), 'res', residual.size(), self.downsample)
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, padding=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1]) #, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2]) #, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3]) #, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet54(ModelBase):
    def __init__(self, num_classes, fs_channels, feature_size=50):
        super().__init__()
        one_channel = 1
        conv_feat_num = 512
        fc_size = int(conv_feat_num * len(fs_channels) * feature_size)
        self.features = nn.ModuleList()

        for fs in fs_channels:
            self.features.append(nn.Sequential(
                ResNet(BasicBlock, [3, 4, 6, 3]),
                #ResNet(Bottleneck, [3, 4, 6, 3]),
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
