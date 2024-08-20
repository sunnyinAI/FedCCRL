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

from data.partition_data import partition_and_statistic, get_partition_arguments, ALL_DOMAINS
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from utils.tools import local_time


def process(test_domain):
    time.sleep(np.random.randint(0, 5))
    # 1. partition data
    data_args = get_partition_arguments()
    data_args.test_domain = test_domain
    dir_name = os.path.join(begin_time, test_domain)
    data_args.directory_name = dir_name
    partition_and_statistic(deepcopy(data_args))
    # 2. train
    fl_args = get_fedavg_argparser().parse_args()
    fl_args.partition_info_dir = dir_name
    fl_args.output_dir = begin_time
    server = FedAvgServer(args=deepcopy(fl_args))
    server.process_classification()


def get_table():
    test_accuracy = {}
    path2dir = os.path.join("out", "FedAvg", "pacs", begin_time)
    for domain in domains:
        with open(os.path.join(path2dir, domain, "test_accuracy.pkl"), "rb") as f:
            test_accuracy[domain] = round(pickle.load(f), 2)
    average_accuracy = round(np.mean(list(test_accuracy.values())), 2)
    test_accuracy["average"] = average_accuracy
    test_accuracy_df = pd.DataFrame(test_accuracy, index=["FedAvg"])
    test_accuracy_df.to_csv(os.path.join(path2dir, "test_accuracy.csv"))
    return test_accuracy


if __name__ == "__main__":
    begin_time = local_time()
    domains = ["photo", "sketch", "art_painting", "cartoon"]
    multiprocess = True
    if multiprocess:
        num_processes = min(len(domains), cpu_count())
        with Pool(processes=num_processes) as pool:
            pool.map(process, domains)
    else:
        for domain in domains:
            process(domain)
    get_table()
