from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Union
import os
import pickle
import torchvision.transforms as transforms

CURRENT_DIR = Path(__file__).parent.absolute()


class FLDataset(Dataset):
    def __init__(
        self,
        args,
        client_id: Union[int, str],
    ):
        self.args = args
        self.client_id = client_id
        client_data_path = os.path.join(
            CURRENT_DIR, args.dataset, args.partition_info_dir, "client_data.pkl"
        )
        with open(client_data_path, "rb") as client_data:
            self.client_data = pickle.load(client_data)
        self.data_paths: List[str] = self.client_data[client_id]["files"]
        self.labels: List[str] = self.client_data[client_id]["labels"]  # label for each sample
        dataset_stats = pickle.load(
            open(
                os.path.join(
                    CURRENT_DIR, args.dataset, args.partition_info_dir, "dataset_stats.pkl"
                ),
                "rb",
            )
        )
        self.label_to_index: Dict[int] = {
            label: idx for idx, label in enumerate(sorted(dataset_stats["label"].keys()))
        }
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        augment_transform = transforms.Compose(
            [
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if args.augment and client_id != "test" and client_id != "validation":
            self.transform = augment_transform
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        # Implement your logic to retrieve and preprocess a single sample from the dataset
        image_path = self.data_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[index]
        label = self.label_to_index[label]
        label = torch.tensor(label)
        return image, label
