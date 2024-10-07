import torch
import torch.nn.functional as F
from torchvision.transforms import AugMix
from torchvision import transforms
import random
from torch import nn

from algorithm.client.fedavg import FedAvgClient
from utils.optimizers_shcedulers import get_optimizer
from utils.tools import trainable_params, get_best_device, local_time
from model.models import MixStyle
import matplotlib.pyplot as plt


class FedCCRLClient(FedAvgClient):
    def __init__(self, args, dataset, client_id, logger):
        super(FedCCRLClient, self).__init__(args, dataset, client_id, logger)
        self.MixStyle = MixStyle(self.args.p, 0.1, self.args.epsilon)

    @torch.no_grad()
    def compute_statistic(self):
        self.move2new_device()
        local_statistic_pool = {"mean": [], "std": []}
        num2upload = int(len(self.train_loader.dataset) * self.args.r)
        batches = int(num2upload / self.args.batch_size)
        left_num = num2upload % self.args.batch_size
        for enu, (data, target) in enumerate(self.train_loader):
            mean = torch.mean(data, dim=(2, 3), keepdim=True)
            var = torch.var(data, dim=(2, 3), keepdim=True)
            std: torch.Tensor = (var + self.args.epsilon).sqrt()
            if enu != batches:
                local_statistic_pool["mean"].append(mean)
                local_statistic_pool["std"].append(std)
            else:
                local_statistic_pool["mean"].append(mean[:left_num])
                local_statistic_pool["std"].append(std[:left_num])
                break
        local_statistic_pool["mean"] = torch.cat(
            local_statistic_pool["mean"], dim=0
        ).to(torch.device("cpu"))
        local_statistic_pool["std"] = torch.cat(local_statistic_pool["std"], dim=0).to(
            torch.device("cpu")
        )
        return local_statistic_pool

    def download_statistic_pool(self, statistic_pool):
        self.statistic_pool = {}
        statistic_pool["mean"].pop(self.client_id)
        statistic_pool["std"].pop(self.client_id)
        statistic_pool["mean"] = [x.to(self.device) for x in statistic_pool["mean"]]
        statistic_pool["std"] = [x.to(self.device) for x in statistic_pool["std"]]
        self.statistic_pool["mean"] = torch.cat(statistic_pool["mean"], dim=0)
        self.statistic_pool["std"] = torch.cat(statistic_pool["std"], dim=0)

    def sample_statistic(self, current_batch_size):
        num = self.statistic_pool["mean"].shape[0]
        if num >= current_batch_size:
            indices = torch.randperm(num)[:current_batch_size]
        else:
            indices = torch.randint(0, num, (current_batch_size,))
        sampled_mean = self.statistic_pool["mean"][indices]
        sampled_std = self.statistic_pool["std"][indices]
        return sampled_mean, sampled_std

    def train(self):
        self.move2new_device()
        self.classification_model.train()
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(self.args.num_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                feature_1 = self.classification_model.base(data)
                output = self.classification_model.classifier(feature_1)
                loss = criterion(output, target)
                output = F.softmax(output, dim=1)
                mix_output = []
                mix_feature = []
                for _ in range(2):
                    mu2, std2 = self.sample_statistic(len(data))
                    generated_data = self.MixStyle(data, mu2, std2)
                    generated_data = self.AugMixAugmentation(generated_data)
                    feature = self.classification_model.base(generated_data)
                    pred = self.classification_model.classifier(feature)
                    loss += criterion(pred, target)
                    if self.args.lambda2 > 0:
                        mix_output.append(F.softmax(pred, dim=1))
                    if self.args.lambda1 > 0:
                        mix_feature.append(feature)

                if self.args.lambda2 > 0:
                    M = torch.clamp(
                        (output + mix_output[0] + mix_output[1]) / 3, 1e-7, 1
                    ).log()
                    kl_1 = F.kl_div(M, output, reduction="batchmean")
                    kl_2 = F.kl_div(M, mix_output[0], reduction="batchmean")
                    kl_3 = F.kl_div(M, mix_output[1], reduction="batchmean")
                    JS_loss = (kl_1 + kl_2 + kl_3) / 3
                    loss += self.args.lambda2 * JS_loss

                if self.args.lambda1 > 0:
                    predication_alignment_loss = (
                        self.supervised_contrastive_loss(
                            mix_feature[0],
                            feature_1,
                            target,
                            temperature=self.args.t,
                        )
                        + self.supervised_contrastive_loss(
                            mix_feature[1],
                            feature_1,
                            target,
                            temperature=self.args.t,
                        )
                    ) / 2
                    loss += self.args.lambda1 * predication_alignment_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
        average_loss = total_loss / len(self.train_loader)
        self.classification_model.to(torch.device("cpu"))
        del self.statistic_pool
        torch.cuda.empty_cache()
        self.logger.log(
            f"{local_time()}, Client {self.client_id}, Avg Loss: {average_loss:.4f}"
        )

    def supervised_contrastive_loss(self, x, y, label, temperature):
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        x = x / x_norm
        y = y / y_norm
        samples = torch.cat((x, y), dim=0)
        label = torch.cat((label, label), dim=0)
        same_label_matrix = torch.eq(label.unsqueeze(1), label.unsqueeze(0)).float()
        sim = torch.matmul(samples, samples.T) / temperature
        same_label_sim = sim * same_label_matrix
        same_label_num = torch.sum(same_label_matrix, dim=1)
        # diff_label_sim = torch.exp(sim) * diff_label_matrix
        negative_sim = torch.exp(sim)
        negative_sum = torch.log(torch.sum(negative_sim, dim=1) - negative_sim.diag())
        positive_sum = torch.sum(same_label_sim, dim=1) / same_label_num
        sum = torch.mean(-positive_sum + negative_sum)
        return sum

    def denormalize(self, tensor, mean, std):
        # Assuming mean and std are lists of channel means and stds
        mean = torch.as_tensor(mean).reshape(1, -1, 1, 1).to(tensor.device)
        std = torch.as_tensor(std).reshape(1, -1, 1, 1).to(tensor.device)
        return tensor * std + mean

    def AugMixAugmentation(self, input_images):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(input_images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(input_images.device)
        input_images = self.denormalize(input_images, mean, std)
        input_images = input_images * 255.0
        input_images = input_images.to(torch.uint8)
        augmix = AugMix()
        # augmixed_images = torch.stack([augmix(x) for x in input_images])
        augmixed_images = augmix(input_images)
        augmixed_images = augmixed_images.float().div(255.0)
        augmixed_images = transforms.Normalize(mean, std)(augmixed_images)
        return augmixed_images

    def scale2unit(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def visualize_augmentation_effect(self, path2dir):
        data_iter = iter(self.train_loader)
        data, _ = next(data_iter)
        random.seed(None)
        random_index = random.randint(0, len(data) - 1)
        sample_image = data[random_index]
        mu2, std2 = self.sample_statistic(1)
        mixstyled_image = self.MixStyle(sample_image.unsqueeze(0), mu2, std2).squeeze(0)
        mixstyled_image = self.scale2unit(mixstyled_image)
        augmixed_image = self.AugMixAugmentation(mixstyled_image.unsqueeze(0)).squeeze(
            0
        )
        sample_image = self.scale2unit(sample_image)
        mixstyled_image = self.scale2unit(mixstyled_image)
        augmixed_image = self.scale2unit(augmixed_image)

        original_image_path = f"{path2dir}/original_image.png"
        plt.imsave(
            original_image_path,
            sample_image.permute(1, 2, 0).cpu().numpy(),
        )
        mixstyle_image_path = f"{path2dir}/mixstyle_image.png"
        plt.imsave(
            mixstyle_image_path,
            mixstyled_image.permute(1, 2, 0).cpu().numpy(),
        )
        augmix_image_path = f"{path2dir}/augmix_image.png"
        plt.imsave(
            augmix_image_path,
            augmixed_image.permute(1, 2, 0).cpu().numpy(),
        )
