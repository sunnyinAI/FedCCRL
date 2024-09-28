from copy import deepcopy
import os
from pathlib import Path
import sys
from argparse import Namespace, ArgumentParser
import pickle
import time
import pandas as pd
from typing import Dict, List
import numpy as np
from rich.console import Console
import torch
from multiprocessing import cpu_count, Pool

# PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
# sys.path.append(PROJECT_DIR.as_posix())

from data.partition_data import (
    partition_and_statistic,
    get_partition_arguments,
    ALL_DOMAINS,
)
from algorithm.server.fedmsfa import FedMSFAServer, get_fedmsfa_argparser
from algorithm.server.fedms import FedMSServer, get_fedms_argparser
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser

mod = "tsne"
assert mod in ["tsne", "augment"]

if mod == "augment":
    server = FedMSServer()
    server.visualize_augmentation_effect()
else:
    args = get_fedavg_argparser().parse_args()
    path_patition_dir = "2024-09-22-20:17:07"
    checkpoint_path = (
        "out/FedMSFA/pacs/num_clients_per_domain_2/AugMix/combine_all/eta_1.0_delta_0.1"
    )
    algo = "FedAvg"
    dataset = "pacs"
    for domain in ALL_DOMAINS[dataset]:
        args.partition_info_dir = os.path.join(path_patition_dir, domain)
        server = FedAvgServer(args=deepcopy(args))
        checkpoint = os.path.join(checkpoint_path, domain, "checkpoint.pth")
        server.resume_checkpoint(checkpoint)
        server.draw_feature_distribution(algo=algo)
