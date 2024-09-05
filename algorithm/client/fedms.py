import torch
import torch.nn.functional as F
import random
from algorithm.client.fedavg import FedAvgClient
from utils.optimizers_shcedulers import get_optimizer
from utils.tools import trainable_params, get_best_device, local_time
from model.models import MixStyle


class FedMSClient(FedAvgClient):
    def __init__(self, args, dataset, client_id, logger):
        super(FedMSClient, self).__init__(args, dataset, client_id, logger)
        self.MixStyle = MixStyle(self.args.p, self.args.mixstyle_alpha, self.args.epsilon)

    @torch.no_grad()
    def compute_statistic(self):
        self.move2new_device()
        local_statistic_pool = {"mean": [], "std": []}
        total_batches = len(self.train_loader)
        assert total_batches * self.args.upload_ratio > 1
        for enu, (data, target) in enumerate(self.train_loader):
            if enu + 1 >= total_batches * self.args.upload_ratio:
                break
            mean = torch.mean(data, dim=(2, 3), keepdim=True)
            var = torch.var(data, dim=(2, 3), keepdim=True)
            std: torch.Tensor = (var + self.args.epsilon).sqrt()
            local_statistic_pool["mean"].append(mean)
            local_statistic_pool["std"].append(std)

        local_statistic_pool["mean"] = torch.cat(local_statistic_pool["mean"], dim=0).to(
            torch.device("cpu")
        )
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
                output = self.classification_model(data)
                loss = criterion(output, target)
                output = F.softmax(output, dim=1)
                mix_output = []
                for _ in range(2):
                    mu2, std2 = self.sample_statistic(len(data))
                    generated_data = self.MixStyle(data, mu2, std2)
                    pred = self.classification_model(generated_data)
                    loss += criterion(pred, target)
                    if self.args.eta > 0:
                        mix_output.append(F.softmax(pred, dim=1))
                if self.args.eta > 0:
                    M = torch.clamp((output + mix_output[0] + mix_output[1]) / 3, 1e-7, 1).log()
                    kl_1 = F.kl_div(M, output, reduction="batchmean")
                    kl_2 = F.kl_div(M, mix_output[0], reduction="batchmean")
                    kl_3 = F.kl_div(M, mix_output[1], reduction="batchmean")
                    JS_loss = (kl_1 + kl_2 + kl_3) / 3
                    loss += self.args.eta * JS_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
        average_loss = total_loss / len(self.train_loader)
        self.classification_model.to(torch.device("cpu"))
        del self.statistic_pool
        torch.cuda.empty_cache()
        self.logger.log(f"{local_time()}, Client {self.client_id}, Avg Loss: {average_loss:.4f}")
