import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * (3840 // 4), 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
