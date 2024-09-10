from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from algorithm.server.fedavg import get_fedavg_argparser
from algorithm.server.fedms import FedMSServer
from algorithm.client.fedmsfa import FedMSFAClient
from data.dataset import FLDataset


def get_fedmsfa_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument("--mixstyle_alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--upload_ratio", type=float, default=0.5)
    parser.add_argument("--p", type=float, default=1)
    parser.add_argument(
        "--eta", type=float, default=40, help="the hyper-parameter for JS divergence"
    )
    parser.add_argument(
        "--delta", type=float, default=1, help="the hyper-parameter for feature alignment loss"
    )
    parser.add_argument(
        "--fa_method",
        type=str,
        choices=["mse", "mi"],
        default="mi",
        help="feature alignment method. 'mi' represents mutual information",
    )
    parser.add_argument("--align2center", type=bool, default=False)
    return parser


class FedMSFAServer(FedMSServer):
    # Federated MixStyle and Feature Alignment
    def __init__(self, algo="FedMSFA", args: Namespace = None):
        if args is None:
            args = get_fedmsfa_argparser().parse_args()
        super().__init__(algo, args)

    def initialize_clients(self):
        self.client_list = [
            FedMSFAClient(self.args, FLDataset(self.args, client_id), client_id, self.logger)
            for client_id in range(self.num_client)
        ]


if __name__ == "__main__":
    server = FedMSFAServer()
    server.process_classification()
