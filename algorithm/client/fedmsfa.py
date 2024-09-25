import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import AugMix
import torchvision.transforms as transforms
import random
from algorithm.client.fedms import FedMSClient
from utils.optimizers_shcedulers import get_optimizer
from utils.tools import trainable_params, get_best_device, local_time
from model.models import MixStyle


class FedMSFAClient(FedMSClient):
    def __init__(self, args, dataset, client_id, logger):
        super(FedMSFAClient, self).__init__(args, dataset, client_id, logger)

    def train(self):
        self.move2new_device()
        self.classification_model.train()
        criterion = torch.nn.CrossEntropyLoss()
        mean, cov = None, None

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
                    if self.args.AugMix:
                        generated_data = self.AugMIxAugmentation(generated_data)
                    feature = self.classification_model.base(generated_data)
                    pred = self.classification_model.classifier(feature)
                    loss += criterion(pred, target)
                    if self.args.eta > 0:
                        mix_output.append(F.softmax(pred, dim=1))
                    if self.args.delta > 0:
                        mix_feature.append(feature)

                if self.args.eta > 0:
                    M = torch.clamp(
                        (output + mix_output[0] + mix_output[1]) / 3, 1e-7, 1
                    ).log()
                    kl_1 = F.kl_div(M, output, reduction="batchmean")
                    kl_2 = F.kl_div(M, mix_output[0], reduction="batchmean")
                    kl_3 = F.kl_div(M, mix_output[1], reduction="batchmean")
                    JS_loss = (kl_1 + kl_2 + kl_3) / 3
                    loss += self.args.eta * JS_loss

                if self.args.delta > 0:
                    if self.args.fa_method == "mse":
                        mse_loss = nn.MSELoss()
                        if self.args.align2center:
                            center = (feature_1 + mix_feature[0] + mix_feature[1]) / 3
                            alignment_loss = (
                                mse_loss(feature_1, center)
                                + mse_loss(mix_feature[0], center)
                                + mse_loss(mix_feature[1], center)
                            ) / 6
                        else:
                            alignment_loss = (
                                mse_loss(feature_1, mix_feature[0])
                                + mse_loss(feature_1, mix_feature[1])
                                + mse_loss(mix_feature[0], mix_feature[1])
                            ) / 6
                        loss += self.args.delta * alignment_loss
                    elif self.args.fa_method == "mi":
                        info_nce_loss = (
                            # self.InfoNCE(mix_feature[0], mix_feature[1])
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
                        loss += self.args.delta * info_nce_loss

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

    @torch.no_grad()
    def mean_cov(self, center=False):
        feature_list = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            feature = self.classification_model.base(data)
            feature_list.append(feature)
            if center:
                for _ in range(2):
                    mu2, std2 = self.sample_statistic(len(data))
                    generated_data = self.MixStyle(data, mu2, std2)
                    feature = self.classification_model.base(generated_data)
                    feature_list.append(feature)
        feature_list = torch.cat(feature_list, dim=0)
        mean = torch.mean(feature_list, dim=0)
        cov = torch.cov(feature_list.T)
        return mean, cov

    def InfoNCE(slef, x, y, temperature=0.1):
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        x = x / x_norm
        y = y / y_norm
        sim_XY = torch.matmul(x, y.T) / temperature
        e_sim_XY = torch.exp(sim_XY)
        diag_e_sim_XY = torch.diag(e_sim_XY)
        sim_XX = torch.matmul(x, x.T) / temperature
        e_sim_XX = torch.exp(sim_XX)
        sim_YY = torch.matmul(y, y.T) / temperature
        e_sim_YY = torch.exp(sim_YY)
        sum_e_sim_xy = torch.sum(e_sim_XY, dim=1)
        sum_e_sim_yx = torch.sum(e_sim_XY, dim=0)
        sum_e_sim_xx = torch.sum(e_sim_XX, dim=1) - e_sim_XX.diag()
        sum_e_sim_yy = torch.sum(e_sim_YY, dim=1) - e_sim_YY.diag()
        g_xy = -torch.log(diag_e_sim_XY / (sum_e_sim_xx + sum_e_sim_xy))
        g_yx = -torch.log(diag_e_sim_XY / (sum_e_sim_yy + sum_e_sim_yx))
        infonce = 0.5 * (g_xy.mean() + g_yx.mean())
        return infonce

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
        if self.args.negative_sample == "diff_label":
            negative_sim = negative_sim * (1 - same_label_matrix)
            negative_sum = torch.log(torch.sum(negative_sim, dim=1))
        else:
            negative_sum = torch.log(
                torch.sum(negative_sim, dim=1) - negative_sim.diag()
            )
        positive_sum = torch.sum(same_label_sim, dim=1) / same_label_num
        sum = torch.mean(-positive_sum + negative_sum)
        return sum
