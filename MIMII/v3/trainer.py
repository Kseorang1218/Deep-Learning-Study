# trainer.py
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score 
import numpy as np
import os

from utils import get_machine_id_list, create_val_file_list, save_model_state_dict, get_filename_list, Visualize_ConfusionMatrix, Get_F1Score
class Trainer:
    def __init__(
            self, model, epoch, device, criterion, optimizer, start_valid_epoch, valid_interval,
            train_dirs, valid_dirs, test_dirs, meta2label_dic, transform, snr):
        self.model = model
        self.epoch = epoch
        self.device = device
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.start_valid_epoch = start_valid_epoch
        self.valid_interval = valid_interval
        self.train_dirs = train_dirs
        self.valid_dirs = valid_dirs
        self.test_dirs = test_dirs
        self.meta2label_dic = meta2label_dic
        self.transform = transform
        self.snr = snr

    def train(self, train_loader):
        best_metric = 0
        start_valid_epoch = self.start_valid_epoch
        valid_interval = self.valid_interval
        epoch = self.epoch
        snr = self.snr
        no_better_epoch = 0
        for epoch in range(0, epoch + 1):
            self.model.train()
            train_loss = 0
            train_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch-{epoch}")
            for (x_wavs, x_mels, labels) in train_bar:
                x_wavs, x_mels = x_wavs.float().to(self.device), x_mels.float().to(self.device)
                labels = labels.reshape(-1).long().to(self.device)
                output, _ = self.model(x_wavs, x_mels, labels)
                loss = self.criterion(output, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            print(f"\n[EPOCH: {epoch}] \tTrain Loss: {avg_train_loss:.4f}")

            # val
            if (epoch - start_valid_epoch) % valid_interval == 0 and epoch >= start_valid_epoch:
                avg_auc, avg_pauc = self.val(epoch)
                if avg_auc + avg_pauc >= best_metric:
                    no_better_epoch = 0
                    best_metric = avg_auc + avg_pauc
                    best_model_path = os.path.join("../model", f'best_checkpoint_{snr}.pth.tar') # TODO: 모델 디렉토리 없으면 만드는 코드 
                    save_model_state_dict(best_model_path, epoch=epoch,
                                                model=self.model,
                                                optimizer=None)
                    print(f'Best epoch now is: {epoch:4d}')
            # if epoch >= self.args.start_save_model_epochs:
            #     if (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
            #         model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
            #         utils.save_model_state_dict(model_path, epoch=epoch,
            #                                     net=self.net.module if self.args.dp else self.net,
            #                                     optimizer=None)
                    
                

    def val(self, epoch=None):
        self.model.eval()
        model = self.model
        sum_auc, sum_pauc, num = 0, 0, 0
        print('\n' + '=' * 20)
        for _, (target_dir, _) in enumerate(zip(sorted(self.valid_dirs), sorted(self.train_dirs))):
            machinetype = target_dir.split("/")[-2]
            machine_id_list = get_machine_id_list(target_dir)
            performance = []
            for modelID in machine_id_list:
                meta = machinetype + '-' + modelID
                label = self.meta2label_dic[meta]
                val_files, y_true = create_val_file_list(target_dir, modelID, dir_name='val')
                y_pred = [0. for _ in val_files]

                for file_idx, file_path in enumerate(val_files):
                    x_wav, x_mel, label = self.transform(file_path)
                    x_wav, x_mel = x_wav.unsqueeze(0).float().to(self.device), x_mel.unsqueeze(0).float().to(self.device)
                    label = torch.tensor([label]).long().to(self.device)
                    with torch.no_grad():
                        predict_ids, _ = model(x_wav, x_mel, label)
                        # print(predict_ids)
                        probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                        # print(probs)
                        y_pred[file_idx] = probs[label]

                # compute auc and pAuc
                max_fpr = 0.1
                auc = roc_auc_score(y_true, y_pred)
                pauc = roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                performance.append([auc, pauc])
        
            avg_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_pauc = avg_performance[0], avg_performance[1]
            print(f"{machinetype}\t\tAUC: {mean_auc*100:.3f}\tpAUC: {mean_pauc*100:.3f}")
            
            sum_auc += mean_auc
            sum_pauc += mean_pauc
            num += 1
        avg_auc, avg_pauc = sum_auc / num, sum_pauc / num
        f1score = Get_F1Score(y_true, y_pred, epoch, f"./saved/{self.snr}/f1scores")
        Visualize_ConfusionMatrix(y_true, y_pred, epoch, f"./saved/{self.snr}/confusionmat")
        print(f'Total average:\t\tAUC: {avg_auc*100:.3f}\tpAUC: {avg_pauc*100:.3f}')
        return avg_auc, avg_pauc
    
    def test(self):
        self.model.eval()
        model = self.model
        print('\n' + '=' * 20)
        for _, (target_dir, _) in enumerate(zip(sorted(self.test_dirs), sorted(self.train_dirs))):
            machinetype = target_dir.split("/")[-2]
            machine_id_list = get_machine_id_list(target_dir)
            for modelID in machine_id_list:
                meta = machinetype + '-' + modelID
                label = self.meta2label_dic[meta]
                test_files = get_filename_list(target_dir, pattern=f'{modelID}*')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel, label = self.transform(file_path)
                    x_wav, x_mel = x_wav.unsqueeze(0).float().to(self.device), x_mel.unsqueeze(0).float().to(self.device)
                    label = torch.tensor([label]).long().to(self.device)
                    with torch.no_grad():
                        predict_ids, _ = model(x_wav, x_mel, label)
                    probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                    y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
