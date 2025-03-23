import torch
from torch.optim import Adam
import numpy as np
import os
import csv

from datetime import datetime
from sklearn.metrics import roc_auc_score

class DeepSVDDTrainer:
    def __init__(self, objective, R, c, nu, config, device):
        self.objective = objective
        self.device = device
        self.config = config

        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

    def train(self, net, train_loader):
        print("\nStarting Training... \n" + "-" * 40)
        net = net.to(self.device)
        optimizer = Adam(net.parameters(), lr=self.config.learning_rate, weight_decay=float(self.config.weight_decay))

        if self.c is None:
            self.c = self.init_center_c(train_loader, net)

        net.train()
        for epoch in range(0, self.config.epoch + 1):
            train_loss = self.training_step(net, train_loader, optimizer, epoch)
            print(f'[EPOCH: {epoch}] \nTrain Loss: {train_loss:.5f}\n')
        return net

    def training_step(self, net, train_loader, optimizer, epoch):
        loss_epoch = 0.0
        n_batches = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            inputs = data.to(self.device)

            optimizer.zero_grad()
            outputs = net(inputs)
            dist = torch.sum((outputs - self.c.unsqueeze(0)) ** 2, dim=1)
            if self.objective == 'soft-boundary':
                scores = dist - self.R ** 2
                loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
            else:
                loss = torch.mean(dist)
            loss.backward()
            optimizer.step()

            if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

            loss_epoch += loss.item()
            n_batches += 1

        return loss_epoch / n_batches
    
    def eval(self, net, eval_loader, latent_size,
             save_result=True, csv_name=None, csv_root=None):
        print("\nStarting Evaluation... \n" + "-" * 40)
        net = net.to(self.device)
        net.eval()
        auc_dic, latent_vectors, fault_label_list = self.validation_step(net, eval_loader)
        for fault, auc in auc_dic.items():
            print(f'{fault} AUC \t{auc:.5f}')

        if save_result:
            if csv_name is None or csv_root is None:
                raise ValueError("Both 'csv_name' and 'csv_root' must be provided when saving the results.")
            self.save_result(csv_name, csv_root, auc_dic, latent_size, self.config.epoch, -1)
        
        return latent_vectors, fault_label_list

    def validation_step(self, net, eval_loader):
        y_true = []
        y_pred = []
        latent_vectors = []

        fault_label_list = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(eval_loader):
                inputs = data.to(self.device)
                label = label.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c.unsqueeze(0)) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist
                
                y_true.append(1 if label.item() > 0 else 0)
                y_pred.append(scores.item())
                fault_label_list.append(label.item())

                latent_vectors.append(outputs.cpu().numpy())

            auc_dic = self.compute_auc(y_true, y_pred, fault_label_list, per_fault=True)

        latent_vectors = np.concatenate(latent_vectors, axis=0)
        return auc_dic, latent_vectors, fault_label_list

    def compute_auc(self, y_true, y_pred, fault_label_list = None, per_fault: bool = True):
        auc_dic = {}
        fault_types = ["normal", "ball", "inner", "outer"]
        
        auc_dic['Total'] = roc_auc_score(y_true, y_pred)

        if per_fault:
            if not fault_label_list:
                raise ValueError("Error: 'fault_label_list' cannot be None when 'per_fault' is True.")
            else:
                for fault in fault_types:
                    if fault == "normal":
                        continue
                    else:
                        fault_indices = [
                            i
                            for i, label in enumerate(fault_label_list)
                            if (label == fault_types.index(fault) or label == 0)
                        ]
                        pred_labels = [y_pred[i] for i in fault_indices]
                        true_labels = [y_true[i] for i in fault_indices]
                        fault_auc = roc_auc_score(true_labels, pred_labels)
                        auc_dic[fault] = fault_auc

        return auc_dic

    def save_result(self, csv_name, csv_root, auc_dic, latent_size, epoch, validation_loss):
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        os.makedirs(f'{csv_root}/', exist_ok=True)
        csv_file = os.path.join(csv_root, f"{csv_name}.csv")

        write_header = not os.path.exists(csv_file)  # 파일이 없으면 헤더를 작성

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            # 첫 번째 줄에 헤더를 추가 (파일이 처음 생성될 때만)
            if write_header:
                writer.writerow(["DateTime", "Latent Size", "Epoch", "Fault", "AUC", "Validation Loss"])

            # 각 fault 별로 행 추가
            for fault, auc in auc_dic.items():
                writer.writerow([current_datetime, latent_size, epoch, fault, auc, validation_loss])


    def init_center_c(self, train_loader, net, eps=0.1):
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(train_loader):
                inputs = data.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
    
def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

