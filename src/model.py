import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .base import ModelBase, TrainingRoutineBase
from .plots import plot_confusion_matrix, plot_cam_1d
from .utils import remove_padding_data, add_padding_data
from .cam1d import GradCAM1D
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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


class ModelCNN(ModelBase):
    def __init__(self, num_classes, fs_channels, input_shape):
        super().__init__()
        one_channel = 1
        self.is_all_fs_same = False

        if all(i == fs_channels[0] for i in fs_channels):
            # all channels has a same sampling frequency
            self.is_all_fs_same = True
            self.features = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(one_channel, 32, kernel_size=5, stride=2),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),

                    nn.Conv1d(32, 64, kernel_size=5, stride=1),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ) for _ in range(len(fs_channels))
            ])

        else:
            # [TODO]
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
            nn.Linear(119296, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = [conv(x[:, i, :, :]) for i, conv in enumerate(self.features)]

        if self.is_all_fs_same:
            # flatten
            x = torch.cat([output.view(output.size(0), -1) for output in x], dim=1)
        else:
            # [TODO]
            #flatten = [remove_padding_data(x[:, i, :, :]).view(batch, -1) for i in range(channel)]
            x = x
        x = self.classifier(x)
        return x

    def __calc_feature_size(self, input_shape):
        # [TODO] NaNでデータ数が減少することを考慮しないといけない、本物のデータをいれるといい
        dummy_input = torch.randn(input_shape)
        with torch.no_grad():
            features_output = self.features(dummy_input)
        return features_output.view(features_output.size(0), -1).size(1)


class ModelTrainingRoutine(TrainingRoutineBase):
    def __init__(self, model, args, criterion='cross_entropy', optimizer='Adam'):
        super().__init__(model, criterion, optimizer, lr=args.lr, gpu_id=args.gpu)
        self.save_itvl = args.save_itvl
        self.test_itvl = args.test_itvl
        self.model_dir = args.model_dir
        self.channel_labels = args.channel_labels
        self.fs_channels = args.fs_channels


    def plot_cam_result(self, inputs, labels, annotation_labels, model_file):
        cams = []
        # set cam targets as the correct label
        targets = [ClassifierOutputTarget(i.item()) for i in labels]
        reload_model = ModelCNN(len(annotation_labels), self.fs_channels, None)

        for feature in reload_model.features:
            reload_model.load_state_dict(torch.load(model_file))
            last_layer = [feature[-1]]
            cam = GradCAM1D(model=reload_model, target_layers=last_layer)
            result = cam(input_tensor=inputs, targets=targets)
            cams.append(result)

        # plot inputs and sequence_cam
        plot_cam_1d(inputs, labels, np.array(cams), annotation_labels, self.channel_labels, self.plot_epoch)
        return


    def run(self, dataset, annotation_labels, channel_labels, num_epoch, batch_size, train_size=0.7):
        train_dataset, test_dataset = dataset.split(size=train_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=os.cpu_count(), # main cpu threads_num
            pin_memory=True,
            drop_last=True,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=False,
        )

        self.model = self.model.to(self.device)

        for epoch in range(num_epoch):
            self.plot_epoch = epoch + 1

            self.model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {self.plot_epoch}/{num_epoch}", dynamic_ncols=True)
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
                self.wandb.update(train_loss=epoch_loss)
            print(f"[LOG] Epoch {self.plot_epoch}/{num_epoch} - Loss: {epoch_loss:.4f}")

            if epoch % self.save_itvl == 0:
                model_file = f"model_e{self.plot_epoch}.pth"
                #torch.save(self.model.state_dict(), os.path.join(self.model_dir, model_file))
                torch.save(self.model.state_dict(),  model_file)
                print(f"[LOG] Model parameters are saved to {model_file}.")

                self.model.eval()
                test_loss = 0.0
                correct = 0
                all_predictions = []
                all_true_labels = []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs).detach()
                        loss = self.criterion(outputs, labels)
                        test_loss += loss.item() * inputs.size(0)
                        _, predicted = outputs.max(1)
                        all_predictions.extend(predicted.cpu().numpy())
                        all_true_labels.extend(labels.cpu().numpy())
                        correct += predicted.eq(labels).sum().item()

                self.plot_cam_result(inputs, labels, annotation_labels, model_file)

                epoch_loss = test_loss / len(test_loader.dataset)
                accuracy = 100. * correct / len(test_loader.dataset)
                if self.wandb is not None:
                    self.wandb.update(test_loss=epoch_loss, acuuracy=accuracy)

                print(f"[LOG] Test Loss after Epoch {self.plot_epoch}: {epoch_loss:.4f}")
                print(f"[LOG] Test Accuracy after Epoch {self.plot_epoch}: {accuracy:.2f}%\n")

                # Compute and plot confusion matrix
                plot_confusion_matrix(np.array(all_true_labels), np.array(all_predictions), classes=annotation_labels, accuracy=accuracy, epoch=self.plot_epoch)
