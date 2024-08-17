import collections
import pickle
from argparse import Namespace

# from collections import OrderedDict

from copy import deepcopy
from typing import Dict, List, Tuple, Union, OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from utils.tools import get_best_device

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
from model.models import get_model_arch
from utils.optimizers import get_optimizer


class FedAvgClient:
    def __init__(self, args, dataset, logger):
        self.args = args
        self.dataset = dataset
        self.logger = logger
        self.classification_model = get_model_arch(self.args.model)(self.args.dataset)
        self.device = get_best_device(self.args.use_cuda)
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)

    def load_model_weights(self, model_weights):
        self.classification_model.load_state_dict(model_weights)

    def train(
        self,
    ):
        self.classification_model.to(self.device)
        self.classification_model.train()
        optimizer = get_optimizer(self.classification_model.parameters(), self.args)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = self.classification_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.logger.info(f"Client {client_id} Epoch: {epoch}, Loss: {total_loss}")
        test_acc = self.test(test_loader)
        return test_acc
