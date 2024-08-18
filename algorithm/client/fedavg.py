from copy import deepcopy
from typing import Dict, List, OrderedDict
from pathlib import Path
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
from model.models import get_model_arch
from utils.optimizers import get_optimizer
from utils.tools import local_time, get_best_device


class FedAvgClient:
    def __init__(self, args, dataset, client_id, logger):
        self.args = args
        self.dataset = dataset
        self.client_id = client_id
        self.logger = logger
        self.classification_model = get_model_arch(model_name=self.args.model)(
            dataset=self.args.dataset
        )
        self.device = get_best_device(self.args.use_cuda)
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)

    def load_model_weights(self, model_weights):
        self.classification_model.load_state_dict(model_weights)

    def get_model_weights(self) -> OrderedDict:
        return self.classification_model.state_dict()

    def train(
        self,
    ):
        self.classification_model.to(self.device)
        self.classification_model.train()
        optimizer = get_optimizer(self.classification_model, self.args)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(self.args.num_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.classification_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        average_loss = total_loss / len(self.train_loader)
        self.logger.log(f"{local_time()}, Client {self.client_id}, Avg Loss: {average_loss:.4f}")
