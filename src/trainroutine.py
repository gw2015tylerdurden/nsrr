import os
import numpy as np
import torch
from tqdm import tqdm
from .base import TrainingRoutineBase
from .plots import plot_confusion_matrix
from .camcaluculator import CamCalculator
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score


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
        self.num_kfolds = args.num_kfolds
        self.kfold = StratifiedKFold(n_splits=args.num_kfolds, shuffle=True, random_state=args.seed)


    def run(self, dataset, args, num_epoch, batch_size, loop):
        train_dataset, train_dataset_indices, test_dataset = dataset.balance_dataset(self.balanced_train_num, self.balanced_test_num)
        test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0 if self.is_debug else os.cpu_count(),
                pin_memory=False if self.is_debug else True,
                drop_last=False,
            )

        def weight_reset(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        best_valid_fold_models = [None] * self.num_kfolds
        for fold, (train_idx, valid_idx) in enumerate(self.kfold.split(list(range(len(train_dataset_indices))), train_dataset_indices)):
            print(f'[LOG] Fold {fold+1}/{self.num_kfolds}')
            self.wandb_init(args, string=f"loop{loop+1}_fold{fold+1}")
            self.model.apply(weight_reset)

            train_subs = Subset(train_dataset, train_idx)
            valid_subs = Subset(train_dataset, valid_idx)

            train_loader = torch.utils.data.DataLoader(
                train_subs,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0 if self.is_debug else os.cpu_count(), # main cpu threads_num
                pin_memory=False if self.is_debug else True,
                drop_last=True,
            )

            valid_loader = torch.utils.data.DataLoader(
                valid_subs,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0 if self.is_debug else os.cpu_count(), # main cpu threads_num
                pin_memory=False if self.is_debug else True,
                drop_last=False,
            )

            self.model = self.model.to(self.device)
            best_valid_score = 0.0
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
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

                # for comfirm overfitting
                self.model.eval()
                true_labels = []
                pred_labels = []
                with torch.no_grad():
                    total_loss_mean = 0.0
                    total_num = 0
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs).detach()
                        loss = self.criterion(outputs, labels)
                        total_loss_mean += loss.item() * inputs.size(0)
                        total_num += inputs.size(0)
                        _, predicted = outputs.max(1)
                        true_labels.extend(labels.cpu().numpy())
                        pred_labels.extend(predicted.cpu().numpy())

                train_acc = accuracy_score(true_labels, pred_labels) * 100
                train_loss = total_loss_mean / total_num

                # validation
                self.model.eval()
                true_labels = []
                pred_labels = []
                with torch.no_grad():
                    total_loss_mean = 0.0
                    total_num = 0
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs).detach()
                        loss = self.criterion(outputs, labels)
                        total_loss_mean += loss.item() * inputs.size(0)
                        total_num += inputs.size(0)
                        _, predicted = outputs.max(1)
                        true_labels.extend(labels.cpu().numpy())
                        pred_labels.extend(predicted.cpu().numpy())

                valid_acc = accuracy_score(true_labels, pred_labels) * 100
                valid_loss = total_loss_mean / total_num
                self.scheduler.step(valid_loss)

                print(f"[LOG] Vlidation Loss after Epoch Epoch {self.plot_epoch}/{num_epoch} - Loss: {valid_loss:.4f}")
                print(f"[LOG] Validation Accuracy after Epoch {self.plot_epoch}: {valid_acc:.2f}%\n")

                if self.wandb is not None:
                    self.wandb.update(train_loss=train_loss,
                                      valid_loss=valid_loss,
                                      train_acc=train_acc,
                                      valid_acc=valid_acc,
                                      )

                if best_valid_score < valid_acc:
                    # remove an old best file
                    os.remove(best_model_file) if best_model_file is not None else None
                    best_valid_score = valid_acc
                    best_model_file = f"model_e{self.plot_epoch}_fold{fold}.pth"
                    best_valid_fold_models[fold] = best_model_file
                    torch.save(self.model.state_dict(),  best_model_file)
                    print(f"[LOG] Model parameters are saved to {best_model_file}.")

            self.wandb.finish()

        # test
        self.model.eval()
        correct = 0
        all_predictions = []
        all_true_labels = []
        with torch.no_grad():
            total_loss_mean = 0.0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs).detach()
                loss = self.criterion(outputs, labels)
                total_loss_mean += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        test_loss = total_loss_mean / len(test_loader.dataset)

        if valid_acc > best_valid_score:
            best_model_file = f"model_e{self.plot_epoch}.pth"
            #torch.save(self.model.state_dict(), os.path.join(self.model_dir, model_file))
            torch.save(self.model.state_dict(),  best_model_file)
            print(f"[LOG] Model parameters are saved to {best_model_file}.")
            best_epoch = self.plot_epoch
            # Compute and plot confusion matrix
            plot_confusion_matrix(np.array(all_true_labels), np.array(all_predictions), classes=self.annotation_labels, accuracy=accuracy, epoch=best_epoch)

        print(f"[LOG] Test Loss after Epoch Epoch {self.plot_epoch}/{num_epoch} - Loss: {test_loss:.4f}")
        print(f"[LOG] Test Accuracy after Epoch {self.plot_epoch}: {accuracy:.2f}%\n")

        # test
        cam_calc = CamCalculator(best_model_file, self.device, self.annotation_labels, self.fs_channels, self.channel_labels, best_epoch)
        cam_calc.plot_cam(test_loader)
        s_min_idx, s_max_idx = cam_calc.calc_sim_result(test_loader, all_predictions)

        return s_min_idx, s_max_idx
