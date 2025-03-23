from torch.optim import Adam
import torch
import numpy as np
import os
import csv

from datetime import datetime
from sklearn.metrics import roc_auc_score

class AETrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def train(self, ae_net, train_loader):
        print("\nStarting pretraining... \n" + "-" * 40)
        ae_net = ae_net.to(self.device)

        optimizer = Adam(ae_net.parameters(), lr = self.config.learning_rate, weight_decay=float(self.config.weight_decay))

        ae_net.train()
        for epoch in range(0, self.config.epoch + 1):
            train_loss = self.training_step(ae_net, train_loader, optimizer)
            print(f'[EPOCH: {epoch}] \nTrain Loss: {train_loss:.5f}\n')

        return ae_net


    def training_step(self, ae_net, train_loader, optimizer):
        loss_epoch = 0.0
        n_batches = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            inputs = data.to(self.device)

            optimizer.zero_grad()

            outputs, _ = ae_net(inputs)
            scores = torch.sum((outputs - inputs) ** 2, dim=(1,2))
            loss = torch.mean(scores)

            loss.backward()
            optimizer.step()


            loss_epoch += loss.item()
            n_batches += 1

        return loss_epoch / n_batches
    
    def eval(self, ae_net, eval_loader, latent_size, 
             save_result=True, csv_name=None, csv_root=None): 
        print("\nTesting Autoencoder... \n" + "-" * 40)
        ae_net = ae_net.to(self.device)
        ae_net.eval()
        loss_epoch, auc_dic = self.validation_step(ae_net, eval_loader)
        print(f'Validation Loss: {loss_epoch:.5f}')
        for fault, auc in auc_dic.items():
            print(f'{fault} AUC \t{auc:.5f}')

        if save_result:
            if csv_name is None or csv_root is None:
                raise ValueError("Both 'csv_name' and 'csv_root' must be provided when saving the results.")
            self.save_result(csv_name, csv_root, auc_dic, latent_size, self.config.epoch, loss_epoch)
            
    def validation_step(self, ae_net, eval_loader):
        y_true = []
        y_pred = []
        latent_vectors = []

        fault_label_list = []
        loss_epoch = 0
        n_batches = 0
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(eval_loader):
                inputs = data.to(self.device)
                label = label.to(self.device)

                output, latent_vector = ae_net(inputs)
                scores = torch.sum((output - inputs) ** 2, dim=(1,2))
                loss = torch.mean(scores)

                loss_epoch += loss.item()
                n_batches += 1

                y_true.append(1 if label.item() > 0 else 0)
                y_pred.append(scores.item())
                fault_label_list.append(label.item())

                latent_vectors.append(latent_vector.cpu().numpy())

            auc_dic = self.compute_auc(y_true, y_pred, fault_label_list, per_fault=True)

        latent_vectors = np.concatenate(latent_vectors, axis=0)

        return loss_epoch/n_batches, auc_dic

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