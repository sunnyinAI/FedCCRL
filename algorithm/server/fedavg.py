from copy import deepcopy
import os
from pathlib import Path
import sys
from argparse import Namespace, ArgumentParser
import pickle
from typing import Dict, List
from rich.console import Console
import torch

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
OUT_DIR = os.path.join(PROJECT_DIR, "out")
sys.path.append(PROJECT_DIR.as_posix())

from utils.tools import (
    fix_random_seed,
    update_args_from_dict,
    local_time,
    Logger,
    get_best_device,
)
from model.models import get_model_arch
from data.partition_data import ALL_DOMAINS
from data.dataset import FLDataset
from algorithm.client.fedavg import FedAvgClient


def get_fedavg_argparser():
    parser = ArgumentParser(description="Fedavg arguments.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pacs", "vlcs", "office_home"],
        default="pacs",
    )
    parser.add_argument(
        "--partition_info_dir",
        type=str,
        default="partition_info",
        help="the name of partition info dir",
    )
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--use-cuda", type=bool, default=True)
    parser.add_argument("--save_log", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--model",
        type=str,
        default="mobile3l",
        choices=["res50", "mobile2", "mobile3s", "mobile3l"],
    )
    parser.add_argument("--augment", type=bool, default=False, help="use data augmentation or not")
    parser.add_argument("--round", type=int, default=10, help="Number of communication rounds")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--test_gap", type=int, default=1)

    return parser


class FedAvgServer:
    def __init__(
        self,
        algo="FedAvg",
        args: Namespace = None,
    ):
        """
        load args & set random seed & create output directory & initialize logger
        """
        self.args = get_fedavg_argparser().parse_args() if args is None else args
        self.algo = algo
        fix_random_seed(self.args.seed)

        with open(
            PROJECT_DIR / "data" / self.args.dataset / self.args.partition_info_dir / "args.pkl",
            "rb",
        ) as f:
            self.args = update_args_from_dict(self.args, pickle.load(f))

        self.path2output_dir = os.path.join(
            OUT_DIR,
            self.algo,
            self.args.dataset,
            self.args.output_dir,
            self.args.test_domain,
        )
        if not os.path.exists(self.path2output_dir):
            os.makedirs(self.path2output_dir)
        self.num_client = (
            len(ALL_DOMAINS[self.args.dataset]) - 1
        ) * self.args.num_clients_per_domain
        self.initialize_logger()
        self.initialize_dataset()
        self.initialize_model()
        self.initialize_clients()

    def initialize_logger(self):
        stdout = Console(log_path=False, log_time=False)
        logfile_path = os.path.join(self.path2output_dir, "log.html")
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.save_log,
            logfile_path=logfile_path,
        )
        self.logger.log("=" * 20, self.algo, self.args.dataset, "=" * 20)
        self.logger.log("Experiment Arguments:", dict(self.args._get_kwargs()))

    def initialize_dataset(self):
        self.test_set = FLDataset(self.args, "test")
        self.validation_set = FLDataset(self.args, "validation")

    def initialize_model(self):
        self.classification_model = get_model_arch(model_name=self.args.model)(
            dataset=self.args.dataset
        )
        self.device = get_best_device(self.args.use_cuda)

    def initialize_clients(self):

        self.client_list = [
            FedAvgClient(self.args, FLDataset(self.args, client_id), client_id, self.logger)
            for client_id in range(self.num_client)
        ]

    def get_agg_weight(self) -> List[float]:
        # Get the weight of each client at the time of aggregation
        num_data_each_client = [len(client.train_loader) for client in self.client_list]
        num_total_data = sum(num_data_each_client)
        weight_list = [num_data / num_total_data for num_data in num_data_each_client]
        return weight_list

    def aggregate_model(self):
        self.agg_weight = self.get_agg_weight()
        model_weight_each_client = [client.get_model_weights() for client in self.client_list]
        new_model_weight = {}
        for key in model_weight_each_client[0].keys():
            new_model_weight[key] = sum(
                [
                    model_weight[key] * weight
                    for model_weight, weight in zip(model_weight_each_client, self.agg_weight)
                ]
            )
        return new_model_weight

    def process_classification(self):
        # distribute original model weights to clients
        for client_id in range(self.num_client):
            self.client_list[client_id].load_model_weights(
                deepcopy(self.classification_model.state_dict())
            )
        self.best_accuracy = 0
        for round_id in range(self.args.round):
            self.logger.log("=" * 20, f"Round {round_id}", "=" * 20)
            for client_id in range(self.num_client):
                self.client_list[client_id].train()
            aggregated_weights = self.aggregate_model()
            self.classification_model.load_state_dict(aggregated_weights)
            for client_id in range(self.num_client):
                self.client_list[client_id].load_model_weights(aggregated_weights)
            if (round_id + 1) % self.args.test_gap == 0:
                self.validate_and_test()

    def validate_and_test(self):
        self.classification_model.eval()
        self.classification_model.to(self.device)
        valid_acc = self.evaluate(self.validation_set)
        self.logger.log(f"{local_time()}, Validation, Accuracy: {valid_acc:.2f}%")
        test_acc = self.evaluate(self.test_set)
        self.logger.log(f"{local_time()}, Test, Accuracy: {test_acc:.2f}%")
        self.classification_model.to(torch.device("cpu"))

        if test_acc > self.best_accuracy:
            self.best_accuracy = test_acc
            test_accuracy_file = os.path.join(self.path2output_dir, "test_accuracy.pkl")
            with open(test_accuracy_file, "wb") as f:
                pickle.dump(test_acc, f)

    def evaluate(self, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size)
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in dataloader:
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.classification_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            accuracy = 100 * correct / total
        return accuracy


if __name__ == "__main__":
    server = FedAvgServer()
    server.process_classification()
