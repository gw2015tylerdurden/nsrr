from sklearn.model_selection import train_test_split
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm



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


def run(annotation_labels, channel_labels, dataset):
    train_dataset, test_dataset = dataset.split(train_size=0.1, random_state=36)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=os.cpu_count(), # main cpu threads_num
        pin_memory=True,
        drop_last=True,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    model = SimpleCNN(num_classes=len(annotation_labels), num_channels=len(channel_labels)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")


def main():
    # https://github.com/nsrr/edf-editor-translator/blob/master/configuration/nsrr-psg-events-compumedics-mapping.csv
    #shhs = ShhsDataLoader(annotation_labels, output_csv=True)
    shhs = ShhsDataLoader(annotation_labels, datasets=['shhs1'], output_csv=False)

    run(annotation_labels, channel_labels, dataset)


if __name__ == '__main__':
    main()
