from copy import deepcopy
import os
from pathlib import Path
import sys
from argparse import Namespace, ArgumentParser
import pickle
from typing import Dict, List
from rich.console import Console
from data.dataset import FLDataset

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
OUT_DIR = os.path.join(PROJECT_DIR, "out")
sys.path.append(PROJECT_DIR.as_posix())

from utils.tools import fix_random_seed, update_args_from_dict, local_time, Logger, get_best_device
from model.models import get_model_arch
from data.partition_data import ALL_DOMAINS
from algorithm.client.fedavg import FedAvgClient


def get_fedavg_arguments():
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
    parser.add_argument("--multi-gpu", type=bool, default=False)
    parser.add_argument("--save_log", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--model",
        type=str,
        default="mobile2",
        choices=["resnet50", "mobile2", "mobile3s", "mobile3s"],
    )
    parser.add_argument("--round", type=int, default=100, help="Number of communication rounds")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    args = parser.parse_args()
    return args


class FedAvgServer:
    def __init__(
        self,
        args: Namespace = None,
    ):
        """
        load args & set random seed & create output directory & initialize logger
        """
        self.args = get_fedavg_arguments().parse_args() if args is None else args
        self.algo = "FedAvg"
        fix_random_seed(self.args.seed_server)
        begin_time = str(local_time())
        self.path2output_dir = (
            OUT_DIR
            / self.algo
            / self.args.dataset
            / self.args.output_dir
            # / (self.args.log_name if self.args.log_name else begin_time)
        )
        if not os.path.exists(self.path2output_dir):
            os.makedirs(self.path2output_dir)

        with open(
            PROJECT_DIR
            / "data"
            / self.args.dataset
            / self.args.preprocessed_file_directory
            / "args.pkl",
            "rb",
        ) as f:
            self.args = update_args_from_dict(self.args, pickle.load(f))

        self.initialize_logger()
        self.initialize_dataset()
        self.initialize_model()

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

        pass

    def initialize_model(self):
        self.classification_model = get_model_arch(self.args.model)(self.args.dataset)
        self.device = get_best_device(self.args.use_cuda)

    def initialize_clients(self):
        self.num_client = (
            len(ALL_DOMAINS[self.args.dataset]) - 1
        ) * self.args.num_clients_per_domain
        self.client_list = [
            FedAvgClient(self.args, FLDataset(self.args, client_id), self.logger)
            for client_id in range(self.num_client)
        ]

    def get_agg_weight(self):
        pass

    def process_classification(self):
        # distribute original model weights to clients
        for client_id in range(self.num_client):
            self.client_list[client_id].load_model_weights(
                deepcopy(self.classification_model.state_dict())
            )
        for round_id in range(self.args.round):
            self.logger.log("=" * 20, f"Round {round_id}" "=" * 20)
            for client_id in range(self.num_client):
                self.client_list[client_id].train()
