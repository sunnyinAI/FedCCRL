import json
import functools
from collections import OrderedDict
from typing import List, Optional
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
import torch.distributions as distributions

NUM_CLASSES = {
    "pacs": 7,
    "vlcs": 5,
    "office_home": 65,
}


def get_model_arch(model_name):
    # static means the model arch is fixed.
    if "fedsr" not in model_name:
        if "res" in model_name:
            return functools.partial(ResNet, version=model_name[3:])
        if "mobile" in model_name:
            return functools.partial(MobileNet, version=model_name[6:])
    else:
        return functools.partial(Model_4_FedSR, base_model_name=model_name.split("_")[-1])


class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: List[nn.Module] = []

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.clone().detach())

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def check_avaliability(self):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.base(x))

    def get_final_features(self, x: Tensor, detach=True) -> Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: Tensor) -> Optional[List[Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list


class ResNet(DecoupledModel):
    def __init__(self, version, dataset: str):
        super().__init__()
        archs = {
            "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        }

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        resnet: models.ResNet = archs[version][0](weights=archs[version][1] if pretrained else None)
        self.base = resnet
        self.classifier = nn.Linear(self.base.fc.in_features, NUM_CLASSES[dataset])
        self.base.fc = nn.Identity()


class MobileNet(DecoupledModel):
    def __init__(self, version, dataset: str):
        super().__init__()
        archs = {
            "2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
            "3s": (
                models.mobilenet_v3_small,
                models.MobileNet_V3_Small_Weights.DEFAULT,
            ),
            "3l": (
                models.mobilenet_v3_large,
                models.MobileNet_V3_Large_Weights.DEFAULT,
            ),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        mobilenet = archs[version][0](weights=archs[version][1] if pretrained else None)
        self.base = mobilenet
        self.classifier = nn.Linear(mobilenet.classifier[-1].in_features, NUM_CLASSES[dataset])
        self.base.classifier[-1] = nn.Identity()
        pass


class Model_4_FedSR(DecoupledModel):
    # modify base model to suit FedSR
    def __init__(self, base_model_name, dataset) -> None:
        super().__init__()
        base_model = get_model_arch(base_model_name)(dataset=dataset)
        self.z_dim = base_model.classifier.in_features
        out_dim = 2 * self.z_dim
        if "mobile" in base_model_name:
            self.base = base_model.base
            # self.classifier = base_model.classifier
            self.base.classifier[-1] = nn.Linear(base_model.classifier.in_features, out_dim)
            self.classifier = nn.Linear(self.z_dim, NUM_CLASSES[dataset])
        elif "res" in base_model_name:
            self.base = base_model.base
            # self.classifier= base_model.classifier
            self.base.fc = nn.Linear(base_model.classifier.in_features, out_dim)
            self.classifier = nn.Linear(self.z_dim, NUM_CLASSES[dataset])
        self.r_mu = nn.Parameter(torch.zeros(NUM_CLASSES[dataset], self.z_dim))
        self.r_sigma = nn.Parameter(torch.ones(NUM_CLASSES[dataset], self.z_dim))
        self.C = nn.Parameter(torch.ones([]))

    def featurize(self, x, num_samples=1, return_dist=False):
        # designed for FedSR
        z_params = self.base(x)
        z_mu = z_params[:, : self.z_dim]
        z_sigma = F.softplus(z_params[:, self.z_dim :])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([num_samples]).view([-1, self.z_dim])

        if return_dist:
            return z, (z_mu, z_sigma)
        else:
            return z

    def forward(self, x):
        z = self.featurize(x)
        logits = self.classifier(z)
        return logits


if __name__ == "__main__":
    model = get_model_arch(model_name="fedsr_res50")(dataset="pacs")
