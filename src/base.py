import torch
from abc import ABC, abstractmethod
from .wandblogging import WandbLogging


class ModelBase(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass


class TrainingRoutineBase(ABC):
    def __init__(self, criterion, device='cuda', gpu_id=0):
        self.model = None
        self.wandb = None

        if criterion == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            print("[ERR] set criterion")
            exit(1)

        self.device = torch.device(device)
        torch.cuda.set_device(gpu_id)

    @abstractmethod
    def run(self):
        pass

    def wandb_init(self, args, string=None):
        self.wandb = WandbLogging(args, string)

    def set_model(self, model, optimizer, lr=1e-3, scheduler='default'):
        self.model = model

        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        else:
            print("[ERR] set optimizer")
            exit(1)
        if scheduler == 'default':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        factor=0.3,
                                                                        patience=2,
                                                                        verbose=True)
        else:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                    gamma=0.9,
                                                                    last_epoch=-1)

