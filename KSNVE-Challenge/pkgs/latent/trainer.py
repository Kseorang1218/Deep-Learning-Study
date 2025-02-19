# trainer.py

import torch
import numpy as np
import os

from sklearn.metrics import roc_auc_score

class Trainer:
    def __init__(self, model, loss, optimizer, device, train_loader, eval_loader):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.eval_loader = eval_loader

    def train(self, epoch):
        print("\nStarting Training... \n" + "-" * 40)
        self.model.train()
        train_loss_list = []
        val_loss_list = []

        for epoch in range(0, epoch + 1):
            train_loss, train_accuracy = self.training_step(self.train_loader)
            print(f'[EPOCH: {epoch}] \nTrain Loss: {train_loss:.5f}\nTrain Acc: {train_accuracy:.5f}\n')
            train_loss_list.append(train_loss)

            _, _, val_loss, _ = self.eval()
            val_loss_list.append(val_loss)

        return train_loss_list, val_loss_list

    def training_step(self, train_loader):
        train_loss = 0.0
        correct = 0

        for batch_idx, (data, label) in enumerate(train_loader):
            x = data.to(self.device)
            label = label.to(self.device)

            output = self.model(x)
            loss = self.loss(output, label.long())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = correct / len(train_loader.dataset)
        return train_loss, train_accuracy
            
    def eval(self):
        print("\nStarting Evaluation... \n" + "-" * 40)
        self.model.eval()
        eval_loss_list, auc_dic, latent_vectors, fault_label_list, predicted_label_list = self.validation_step(self.eval_loader)

        print(f'Validation Loss: {np.mean(eval_loss_list):.5f}')
        for fault, auc in auc_dic.items():
            print(f'{fault} AUC \t{auc:.5f}')
        print()
            
        return latent_vectors, fault_label_list, np.mean(eval_loss_list), predicted_label_list


    def validation_step(self, eval_loader):
        eval_loss_list = []

        y_true = []
        y_pred = []
        latent_vectors = []

        fault_label_list = []
        predicted_label_list = []

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(eval_loader):
                x = data.to(self.device)
                label = label.to(self.device)

                output = self.model(x)
                loss = self.loss(output, label.long())

                eval_loss_list.append(loss.item())

                y_true.append(1 if label.item() > 0 else 0)
                y_pred.append(loss.item())

                prediction = output.max(1, keepdim = True)[1]
                fault_label_list.append(label.item())
                predicted_label_list.append(prediction.item())

                latent_vectors.append(output.cpu().numpy())

            auc_dic = self.compute_auc(y_true, y_pred, fault_label_list, per_fault=True)

        latent_vectors = np.concatenate(latent_vectors, axis=0)

        return eval_loss_list, auc_dic, latent_vectors, fault_label_list, predicted_label_list

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

    def save(self, root, latent_size = False):
        os.makedirs(f'{root}/', exist_ok=True)

        if latent_size:     
            # latent space 실험용         
            torch.save(self.model.state_dict(), f'{root}/model_{latent_size}.pt')
        else:
            # 그냥 저장 
            torch.save(self.model.state_dict(), f'{root}/model.pt')