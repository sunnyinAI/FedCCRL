from copy import deepcopy
import os
from pathlib import Path
import random
import sys
from argparse import Namespace, ArgumentParser
import pickle
import time
from matplotlib import pyplot as plt
import pandas as pd
from typing import Dict, List
import numpy as np
from rich.console import Console
import torch
from torchvision.io import read_image
import torchvision.transforms as transforms

from multiprocessing import cpu_count, Pool
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.tools import (
    fix_random_seed,
    update_args_from_dict,
    local_time,
    Logger,
    get_best_device,
)
from model.models import get_model_arch
from torchvision.io import ImageReadMode

# PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
# sys.path.append(PROJECT_DIR.as_posix())

from data.partition_data import (
    partition_and_statistic,
    get_partition_arguments,
    ALL_DOMAINS,
)
from data.dataset import zero_one_normalize
from algorithm.server.fedmsfa import FedMSFAServer, get_fedmsfa_argparser
from algorithm.server.fedms import FedMSServer, get_fedms_argparser
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser

mod = "grad-cam"
assert mod in ["tsne", "augment", "grad-cam"]

if mod == "augment":
    server = FedMSServer()
    server.visualize_augmentation_effect()
elif mod == "grad-cam":
    dataset = "pacs"
    domain = "sketch"
    model = get_model_arch(model_name="mobile3l")(dataset=dataset)
    fedavg_checkpoint = f"out/FedAvg/pacs/2024-09-28-15:43:20/{domain}/checkpoint.pth"
    fedmsfa_checkpoint = f"out/FedMSFA/pacs/num_clients_per_domain_2/AugMix/combine_all/eta_1.0_delta_0.1/{domain}/checkpoint.pth"
    dataset_path = os.path.join("data", dataset, "raw", domain)
    labels = os.listdir(dataset_path)
    for i in range(100):
        random_label = random.choice(labels)
        image_path = os.path.join(
            dataset_path,
            random_label,
            random.choice(os.listdir(os.path.join(dataset_path, random_label))),
        )

        original_image = read_image(image_path)
        transform = transforms.Compose(
            [
                (transforms.Resize((224, 224))),
                zero_one_normalize(),
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = transform(original_image)
        image = image.unsqueeze(0)
        model = model.cuda()
        input_image = image.cuda()
        model.load_state_dict(torch.load(fedavg_checkpoint, weights_only=True)["model"])
        model.eval()
        fedavg_prediction = model(input_image).argmax(dim=1).item()

        model.load_state_dict(
            torch.load(fedmsfa_checkpoint, weights_only=True)["model"]
        )
        model.eval()
        fedmsfa_prediction = model(input_image).argmax(dim=1).item()
        label_index = labels.index(random_label)
        # Check if predictions are different and save the image if they are
        if fedavg_prediction != fedmsfa_prediction and fedavg_prediction == label_index:
            print(random_label, fedavg_prediction, fedmsfa_prediction)
            path2save = os.path.join("image", "grad_cam", local_time())
            if not os.path.exists(path2save):
                os.makedirs(path2save)
            model.load_state_dict(
                torch.load(fedavg_checkpoint, weights_only=True)["model"]
            )
            model.eval()
            target_layer = model.base.features[-1]
            # Save the original image
            original_image_path = os.path.join(path2save, "original_image.png")
            plt.imsave(original_image_path, original_image.permute(1, 2, 0).numpy())

            cam = GradCAM(model=model, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=input_image)
            image = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image = image - image.min()
            image = image / image.max()
            cam_image = show_cam_on_image(image, grayscale_cam[0], use_rgb=True)
            grad_cam_image_path = os.path.join(path2save, "fedavg.png")
            plt.imsave(grad_cam_image_path, cam_image)

            model.load_state_dict(
                torch.load(fedmsfa_checkpoint, weights_only=True)["model"]
            )
            model.eval()
            target_layer = model.base.features[-1]
            cam = GradCAM(model=model, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=input_image)
            cam_image = show_cam_on_image(image, grayscale_cam[0], use_rgb=True)
            plt.imsave(os.path.join(path2save, "fedmsfa.png"), cam_image)
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
