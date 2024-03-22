import os
import numpy as np
import torch
from tqdm import tqdm
from .base import TrainingRoutineBase
from .plots import plot_confusion_matrix
from .camcaluculator import CamCalculator


class ModelTrainingRoutine(TrainingRoutineBase):
    def __init__(self, model, fs_channels, channels, annotation_labels, args, criterion='cross_entropy', optimizer='Adam'):
        super().__init__(model, criterion, optimizer, lr=args.lr, gpu_id=args.gpu)
        self.save_itvl = args.save_itvl
        self.test_itvl = args.test_itvl
        self.model_dir = args.model_dir
        self.is_debug = args.is_debug
        self.balanced_train_num = args.balanced_train_num
        self.balanced_test_num = args.balanced_test_num

        self.channel_labels = channels
        self.fs_channels = fs_channels
        self.annotation_labels = annotation_labels


    def run(self, dataset, num_epoch, batch_size, train_size=0.7):
        train_dataset, test_dataset = dataset.balance_dataset(self.balanced_train_num, self.balanced_test_num)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 if self.is_debug else os.cpu_count(), # main cpu threads_num
            pin_memory=False if self.is_debug else True,
            drop_last=True,
        )

        validation_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0 if self.is_debug else os.cpu_count(),
            pin_memory=False if self.is_debug else True,
            drop_last=False,
        )

        self.model = self.model.to(self.device)
        best_validation_score = 0.0
        best_model_file = None
        best_epoch = 0

        for epoch in range(num_epoch):
            self.plot_epoch = epoch + 1

            self.model.train()
            total_loss_mean = 0.0
            total_num = 0
            pbar = tqdm(train_loader, desc=f"Epoch {self.plot_epoch}/{num_epoch}", dynamic_ncols=True)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                total_loss_mean += loss.item() * inputs.size(0)
                total_num += inputs.size(0)

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if self.wandb is not None:
                self.wandb.update(train_loss=total_loss_mean / total_num)
            print(f"[LOG] Epoch {self.plot_epoch}/{num_epoch} - Loss: {total_loss_mean / total_num:.4f}")


            if epoch % self.save_itvl == 0:
                model_file = f"model_e{self.plot_epoch}.pth"
                #torch.save(self.model.state_dict(), os.path.join(self.model_dir, model_file))
                torch.save(self.model.state_dict(),  model_file)
                print(f"[LOG] Model parameters are saved to {model_file}.")

                self.model.eval()
                correct = 0
                all_predictions = []
                all_true_labels = []
                with torch.no_grad():
                    total_loss_mean = 0.0
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs).detach()
                        loss = self.criterion(outputs, labels)
                        total_loss_mean += loss.item() * inputs.size(0)
                        _, predicted = outputs.max(1)
                        all_predictions.extend(predicted.cpu().numpy())
                        all_true_labels.extend(labels.cpu().numpy())
                        correct += predicted.eq(labels).sum().item()

                accuracy = 100. * correct / len(validation_loader.dataset)
                if self.wandb is not None:
                    self.wandb.update(test_loss=total_loss_mean / len(validation_loader.dataset), acuuracy=accuracy)

                if accuracy > best_validation_score:
                    best_validation_score = accuracy
                    best_model_file = model_file
                    best_epoch = self.plot_epoch
                    # Compute and plot confusion matrix
                    plot_confusion_matrix(np.array(all_true_labels), np.array(all_predictions), classes=self.annotation_labels, accuracy=accuracy, epoch=best_epoch)

                print(f"[LOG] Test Loss after Epoch Epoch {self.plot_epoch}/{num_epoch} - Loss: {total_loss_mean / len(validation_loader.dataset):.4f}")
                print(f"[LOG] Test Accuracy after Epoch {self.plot_epoch}: {accuracy:.2f}%\n")

        if self.wandb is not None:
            self.wandb.update(best_iteration_accuracy=best_validation_score)

        cam_calc = CamCalculator(best_model_file, self.device, self.annotation_labels, self.fs_channels, self.channel_labels, best_epoch)
        cam_calc.plot_cam(validation_loader)
        s_min_idx, s_max_idx = cam_calc.calc_sim_result(validation_loader, all_predictions)

        return s_min_idx, s_max_idx
