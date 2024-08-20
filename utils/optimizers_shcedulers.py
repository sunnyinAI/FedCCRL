import torch
import torch.optim as optim


def get_optimizer(model, args):
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def get_scheduler():
    return CosineAnnealingLRWithWarmup
    pass


class CosineAnnealingLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs: int, last_epoch: int = -1):
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Calculate the current learning rate using cosine annealing
        return [
            base_lr
            * 0.5
            * (1 + torch.cos(torch.tensor(self.last_epoch / self.total_epochs * torch.pi)))
            for base_lr in self.base_lrs
        ]
