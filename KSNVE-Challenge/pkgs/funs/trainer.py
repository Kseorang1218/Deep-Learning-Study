# trainer.py

import torch
import numpy as np
import os
import csv

from sklearn.metrics import roc_auc_score

class Trainer:
    def __init__(self, model, loss, optimizer, device):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device

    def train(self, epoch, train_loader):
        print("\nStarting Training... \n" + "-" * 40)
        self.model.train()
        for epoch in range(0, epoch + 1):
            train_loss_list = self.training_step(train_loader)
            print(f'[EPOCH: {epoch}] \nTrain Loss: {np.mean(train_loss_list):.5f}\n')

    def training_step(self, train_loader):
        train_loss_list = []

        for batch_idx, (data, label) in enumerate(train_loader):
            x = data.to(self.device)
            output, _ = self.model(x)
            loss = self.loss(x, output)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss_list.append(loss.item())

        return train_loss_list
            
    def eval(self, eval_loader, csv_name, latent_size):
        print("\nStarting Evaluation... \n" + "-" * 40)
        self.model.eval()
        eval_loss_list, auc_dic, latent_vectors, fault_label_list = self.validation_step(eval_loader)
        print(f'Validation Loss: {np.mean(eval_loss_list):.5f}')
        for fault, auc in auc_dic.items():
            print(f'{fault} AUC \t{auc:.5f}')

        with open(f"{csv_name}.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            # 첫 번째 줄에는 헤더를 추가하려면 파일이 비었을 때만 추가
            writer.writerow(["Fault", "AUC", "Validation Loss", "latent size"])  # CSV 헤더 작성
            for fault, auc in auc_dic.items():
                writer.writerow([fault, auc, np.mean(eval_loss_list), latent_size])  # Validation Loss를 각 AUC 항목과 함께 기록
            
        return latent_vectors, fault_label_list


    def validation_step(self, eval_loader):
        eval_loss_list = []

        y_true = []
        y_pred = []
        latent_vectors = []

        fault_label_list = []

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(eval_loader):
                x = data.to(self.device)
                label = label.to(self.device)

                output, latent_vector = self.model(x)
                loss = self.loss(x, output)

                eval_loss_list.append(loss.item())

                y_true.append(1 if label.item() > 0 else 0)
                y_pred.append(loss.item())
                fault_label_list.append(label.item())

                latent_vectors.append(latent_vector.cpu().numpy())

            auc_dic = self.compute_auc(y_true, y_pred, fault_label_list, per_fault=True)

        latent_vectors = np.concatenate(latent_vectors, axis=0)

        return eval_loss_list, auc_dic, latent_vectors, fault_label_list

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

    def save(self, root, model_name, latent_size = False):
        os.makedirs(f'{root}/', exist_ok=True)

        if latent_size:     
            # latent space 실험용         
            torch.save(self.model.state_dict(), f'{root}/{model_name}_{latent_size}.pt')
        else:
            # 그냥 저장 
            torch.save(self.model.state_dict(), f'{root}/{model_name}.pt')