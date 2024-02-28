import os
import numpy as np
import torch
from tqdm import tqdm
from .base import TrainingRoutineBase
from .plots import plot_confusion_matrix
from .camcaluculator import CamCalculator


class ModelTrainingRoutine(TrainingRoutineBase):
    def __init__(self, model, args, criterion='cross_entropy', optimizer='Adam'):
        super().__init__(model, criterion, optimizer, lr=args.lr, gpu_id=args.gpu)
        self.save_itvl = args.save_itvl
        self.test_itvl = args.test_itvl
        self.model_dir = args.model_dir
        self.channel_labels = args.channel_labels
        self.fs_channels = args.fs_channels
        self.threshold_calc_sim_epochs = args.threshold_calc_sim_epochs
        self.is_debug = args.is_debug


    def run(self, dataset, annotation_labels, channel_labels, num_epoch, batch_size, train_size=0.7):
        train_dataset, test_dataset = dataset.balance_dataset(train_data_num=2000,
                                                              test_data_num=1000)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if self.is_debug else os.cpu_count(), # main cpu threads_num
            pin_memory=False if self.is_debug else True,
            drop_last=True,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0 if self.is_debug else os.cpu_count(),
            pin_memory=False if self.is_debug else True,
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
                cam_calc = CamCalculator(model_file, self.device, annotation_labels, self.fs_channels, self.channel_labels, self.plot_epoch)
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

                epoch_loss = test_loss / len(test_loader.dataset)
                accuracy = 100. * correct / len(test_loader.dataset)
                if self.wandb is not None:
                    self.wandb.update(test_loss=epoch_loss, acuuracy=accuracy)

                print(f"[LOG] Test Loss after Epoch {self.plot_epoch}: {epoch_loss:.4f}")
                print(f"[LOG] Test Accuracy after Epoch {self.plot_epoch}: {accuracy:.2f}%\n")

                #if epoch >= self.threshold_calc_sim_epochs:
                cam_calc.plot_sim_result(test_loader, all_predictions)
                cam_calc.plot_cam(test_loader)

                # Compute and plot confusion matrix
                plot_confusion_matrix(np.array(all_true_labels), np.array(all_predictions), classes=annotation_labels, accuracy=accuracy, epoch=self.plot_epoch)
