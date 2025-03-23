import torch
from torch import optim
import torch.nn as nn
import numpy as np
import os
import csv

from datetime import datetime

from sklearn.metrics import roc_auc_score

def weights_init_normal(m):
    # if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):  # 가중치가 있는 레이어만 초기화
    #     torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    pass

class TrainerDeepSVDD:
    def __init__(self, pre_model, model, loss, optimizer, device, train_loader, eval_loader, is_pretrain):
        self.pre_model = pre_model
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.is_pretrain = is_pretrain


    def train(self, epoch, train_loader, latent_dim):
        print("\nStarting Training... \n" + "-" * 40)
        self.model.train()

        train_loss_list = []

        if self.is_pretrain==True:
            state_dict = torch.load('weights/pretrained_parameters.pth')
            self.model.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            self.model.apply(weights_init_normal)
            c = torch.randn(latent_dim).to(self.device)

        for epoch in range(0, epoch + 1):
            train_loss, train_accuracy = self.training_step(train_loader, c)
            print(f'[EPOCH: {epoch}] \nTrain Loss: {train_loss:.5f}\nTrain Acc: {train_accuracy:.5f}\n')
            train_loss_list.append(train_loss)


        return train_loss_list
    

    def training_step(self, train_loader, c):
        train_loss = 0.0
        correct = 0

        for batch_idx, (data, label) in enumerate(train_loader):
            x = data.to(self.device)
            label = label.to(self.device)

            output = self.model(x)
            loss = torch.mean(torch.sum((output - c) ** 2, dim=1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            prediction = output.max(1, keepdim = True)[1]
        correct += prediction.eq(label.view_as(prediction)).sum().item()
    
        train_loss /= len(train_loader)
        train_accuracy = correct / len(train_loader.dataset)

        self.c = c

        return train_loss, train_accuracy
    

    def pretrain(self, train_loader, epoch):
        train_loss = 0.0
        self.pre_model.apply(weights_init_normal)

        optimizer = optim.Adam(self.pre_model.parameters(), 0.001)
        
        self.pre_model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            x = data.to(self.device)
            
            optimizer.zero_grad()
            x_hat, _ = self.pre_model(x)
            reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
            reconst_loss.backward()
            optimizer.step()
            
            train_loss += reconst_loss.item()
        print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, train_loss/len(train_loader)))
        self.save_weights_for_DeepSVDD(self.pre_model, train_loader) 
    

    def save_weights_for_DeepSVDD(self, model, dataloader):
        c = self.set_c(model, dataloader)
        net = self.model().to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, 'weights/pretrained_parameters.pth')
    

    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(dataloader):
                x = data.to(self.device)
                z = model.encoder(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    
    def eval(self, eval_loader, latent_size, epoch, save_result=True, csv_name=None, csv_root=None):
        print("\nStarting Evaluation... \n" + "-" * 40)

        self.model.eval()
        eval_loss_list, auc_dic, latent_vectors, fault_label_list = self.validation_step(eval_loader)

        print(f'Validation Loss: {np.mean(eval_loss_list):.5f}')
        for fault, auc in auc_dic.items():
            print(f'{fault} AUC \t{auc:.5f}')

        if save_result:
            if csv_name is None or csv_root is None:
                raise ValueError("Both 'csv_name' and 'csv_root' must be provided when saving the results.")
            self.save_result(csv_name, csv_root, auc_dic, latent_size, epoch, np.mean(eval_loss_list))
            
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

                output = self.model(x)
                score = torch.sum((output - self.c) ** 2, dim=1)

                eval_loss_list.append(score.item())

                y_true.append(1 if label.item() > 0 else 0)
                y_pred.append(score.item())
                fault_label_list.append(label.item())

                latent_vectors.append(output.cpu().numpy())

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
    

        

