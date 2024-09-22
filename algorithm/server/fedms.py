from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
import sys

import torch

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.client.fedms import FedMSClient
from data.dataset import FLDataset


def get_fedms_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument("--mixstyle_alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--upload_ratio", type=float, default=0.5)
    parser.add_argument("--p", type=float, default=1)
    parser.add_argument(
        "--eta", type=float, default=0.001, help="the hyper-parameter for JS divergence"
    )
    parser.add_argument("--AugMix", type=bool, default=True)
    # parser.add_argument("--k", type=int, default=2)
    return parser


class FedMSServer(FedAvgServer):
    def __init__(self, algo="FedMS", args: Namespace = None):
        if args is None:
            args = get_fedms_argparser().parse_args()
        super().__init__(algo, args)

    def initialize_clients(self):
        self.client_list = [
            FedMSClient(
                self.args, FLDataset(self.args, client_id), client_id, self.logger
            )
            for client_id in range(self.num_client)
        ]

    def generate_statistic_pool(self):
        # generate style pool
        statistic_pool = {"mean": [], "std": []}
        for client_id in range(self.num_client):
            local_statistic_pool = self.client_list[client_id].compute_statistic()
            statistic_pool["mean"].append(local_statistic_pool["mean"])
            statistic_pool["std"].append(local_statistic_pool["std"])
            torch.cuda.empty_cache()

        return statistic_pool

    def process_classification(self):
        # distribute original model weights to clients
        for client_id in range(self.num_client):
            self.client_list[client_id].load_model_weights(
                deepcopy(self.classification_model.state_dict())
            )

        self.best_accuracy = 0
        for round_id in range(self.args.round):
            self.logger.log("=" * 20, f"Round {round_id}", "=" * 20)
            # generate style pool
            statistic_pool = self.generate_statistic_pool()
            for client_id in range(self.num_client):
                self.client_list[client_id].download_statistic_pool(
                    deepcopy(statistic_pool)
                )
                self.client_list[client_id].train()

            aggregated_weights = self.aggregate_model()
            self.classification_model.load_state_dict(aggregated_weights)
            for client_id in range(self.num_client):
                self.client_list[client_id].load_model_weights(aggregated_weights)
            if (round_id + 1) % self.args.test_gap == 0:
                self.validate_and_test()
            self.save_checkpoint(round_id)


if __name__ == "__main__":
    server = FedMSServer()
    server.process_classification()
