import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import src.seeds as DeterministicSeed
from src.shhsdataload import ShhsDataLoader
import src.model as SimpleCNN
from tqdm import tqdm
import os

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



@hydra.main(config_path="config", config_name="parameters")
def main(args):
    shhs = ShhsDataLoader(args.annotation_labels,
                          base_path=args.original_dataset_path,
                          datasets=args.datasets,
                          output_csv=args.output_shhs_datainfo_csv)

    shhs.create_target_fs_dataset_h5(args.channel_labels,
                                     target_fs=None,
                                     creation_filename=os.path.expanduser(args.creation_dataset_name),
                                     debug_plots_interval=args.debug_plots_interval)
    #run(annotation_labels, channel_labels, dataset)


if __name__ == '__main__':
    main()
