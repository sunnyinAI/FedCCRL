import os
from pathlib import Path
import pickle
from typing import Dict, List, Union
from matplotlib import pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import argparse
import sys
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent.parent.absolute()
CURRENT_DIR = Path(__file__).parent.absolute()

sys.path.append(PROJECT_DIR.as_posix())
from utils.heterogeneity import heterogeneity

ALL_DOMAINS = {
    "pacs": ["art_painting", "cartoon", "photo", "sketch"],
    "vlcs": ["caltech", "labelme", "pascal", "voc"],
    "office_home": ["art", "clipart", "product", "realworld"],
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Federated Domain Generalization with PACS Dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pacs",
        choices=["pacs", "vlcs", "office_home"],
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test_domain", type=str, default="cartoon", help="Test domain")
    parser.add_argument(
        "--num_clients_per_domain",
        type=int,
        default=2,
        help="Number of clients for each domain",
    )
    parser.add_argument(
        "--hetero_method", type=str, default="dirichlet", help="Heterogeneity method"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value for Dirichlet heterogeneity"
    )
    args = parser.parse_args()
    return args


def partition_data(args) -> Dict[Union[int, str], Dict[str, List]]:
    """
    Summary:
        Generate data partition for federated learning.
        Validation dataset is extracted from training data with ratio 10%.

    Args:
        args (_type_): _description_

    Returns:
        Dict[Union[int, str], Dict[str, List]]:
            keys: client_id, 'validation', 'test'
            values: Dict with keys 'files', 'labels', 'domain', values are list
                'files': list of file paths
                'labels': list of labels
                'domain': list of domain names
    """
    domains = ALL_DOMAINS[args.dataset]
    domain_paths = {
        domain: os.path.join(PROJECT_DIR, f"data/{args.dataset}/raw", domain) for domain in domains
    }
    test_domain = args.test_domain
    client_domains = [domain for domain in domains if domain != test_domain]
    num_clients = args.num_clients_per_domain * len(client_domains)
    all_files = defaultdict(list)
    all_labels = defaultdict(list)
    for domain, path in domain_paths.items():
        for cls in os.listdir(path):
            cls_path = os.path.join(path, cls)
            files = os.listdir(cls_path)
            all_files[domain].extend([os.path.join(cls_path, f) for f in files])
            all_labels[domain].extend([cls] * len(files))

    domain_distribution = heterogeneity(args, client_domains)
    # partition training data
    client_data = defaultdict(dict)
    client_data["validation"] = defaultdict(list)
    for client_id in range(num_clients):
        client_data[client_id] = defaultdict(list)
    for domain, assignment in domain_distribution.items():
        n = len(all_files[domain])  # Number of files in this domain
        indices = np.arange(n)
        np.random.shuffle(indices)
        split_idx = int(n * 0.9)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        num_training_data = len(train_indices)
        num_elements_gt_zero = sum(1 for val in assignment if val > 0)
        current_idx = 0
        for client_id, proportion in enumerate(assignment):
            # Distribute files to client assigned with this domain
            if proportion > 0:
                if num_elements_gt_zero != 1:
                    # if this client is not the last client with non-zero proportion
                    num_samples = int(proportion * num_training_data)
                else:
                    # if is the last, assign all remaining samples
                    num_samples = num_training_data - current_idx + 1
                client_data_indices = train_indices[current_idx : current_idx + num_samples]
                client_data[client_id]["files"].extend(
                    [all_files[domain][i] for i in client_data_indices]
                )
                client_data[client_id]["labels"].extend(
                    [all_labels[domain][i] for i in client_data_indices]
                )
                client_data[client_id]["domain"].extend(domain)
                current_idx += num_samples
                num_elements_gt_zero -= 1
        client_data["validation"]["files"].extend([all_files[domain][i] for i in val_indices])
        client_data["validation"]["labels"].extend([all_labels[domain][i] for i in val_indices])
        client_data["validation"]["domain"].extend(domain)

    # data for test and validation
    client_data["test"] = {
        "files": all_files[test_domain],
        "labels": all_labels[test_domain],
        "domain": [test_domain],
    }
    return client_data


def count_samples_per_domain(client_data):
    domain_counts = defaultdict(lambda: defaultdict(int))
    for client_id, data in client_data.items():
        if isinstance(client_id, int):  # Skip 'validation' and 'test'
            for domain in data["domain"]:
                domain_counts[client_id][domain] += 1
    return domain_counts


def count_labels_per_client(client_data):
    label_counts = defaultdict(lambda: defaultdict(int))
    for client_id, data in client_data.items():
        if isinstance(client_id, int):  # Skip 'validation' and 'test'
            for label in data["labels"]:
                label_counts[client_id][label] += 1
    return label_counts


def plot_sample_distribution(domain_counts):
    clients = sorted(domain_counts.keys())
    domains = sorted({domain for counts in domain_counts.values() for domain in counts.keys()})

    # Data preparation for plotting
    data = {domain: [domain_counts[client][domain] for client in clients] for domain in domains}

    fig, ax = plt.subplots()

    for domain, counts in data.items():
        ax.bar(clients, counts, label=domain)

    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Sample Distribution Across Clients")
    ax.legend(title="Domain")
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    np.random.seed(args.seed)
    test_domain = args.test_domain
    all_domains = ALL_DOMAINS[args.dataset]
    assert test_domain in all_domains, f"Test domain {test_domain} not found in {args.dataset}"
    client_data = partition_data(args)
