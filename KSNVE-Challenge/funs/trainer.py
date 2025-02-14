# trainer.py

import torch
import numpy as np
import os

from sklearn.metrics import roc_auc_score

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
from sklearn.decomposition import PCA

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
            
    def eval(self, eval_loader):
        print("\nStarting Evaluation... \n" + "-" * 40)
        self.model.eval()
        eval_loss_list, auc_dic, latent_vectors, fault_label_list = self.validation_step(eval_loader)
        print(f'Validation Loss: {np.mean(eval_loss_list):.5f}')
        for fault, auc in auc_dic.items():
            print(f'{fault} AUC \t{auc:.5f}')

        # tsne = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto')
        # latent_vectors_tsne = tsne.fit_transform(latent_vectors)

        # plt.figure(figsize=(10, 8))
        # fault_label_list = np.array(fault_label_list, dtype=int)

        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # fault_label_list = np.array(fault_label_list, dtype=int)

        
        # # 클래스별 텍스트 추가
        # fault_types = ["normal", "ball", "inner", "outer"]
        # scatter = {}
        # for i, fault in enumerate(fault_types):
        #     if i == 2:
        #         continue
        #     else:
        #         class_indices = np.where(fault_label_list == i)[0]
        #         scatter[fault] = ax.scatter(latent_vectors_tsne[class_indices, 0], 
        #                                     latent_vectors_tsne[class_indices, 1], 
        #                                     latent_vectors_tsne[class_indices, 2], 
        #                                     label=fault, alpha=0.5, s=10)  # 점 크기 조정

        # # 범례 추가
        # ax.set_title("3D t-SNE Visualization of Latent Vectors")
        # ax.set_xlabel("t-SNE Component 1")
        # ax.set_ylabel("t-SNE Component 2")
        # ax.set_zlabel("t-SNE Component 3")
        # ax.legend(title="Class Labels")  # 범례 제목
        # plt.savefig('./tmp_3d.png')

        # tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        # latent_vectors_tsne = tsne.fit_transform(latent_vectors)

        # plt.figure(figsize=(10, 8))
        # fault_label_list = np.array(fault_label_list, dtype=int)

        # scatter = {}
        # for i, fault in enumerate(fault_types):
        #     if i == 2:
        #         continue
        #     else:
        #         class_indices = np.where(fault_label_list == i)[0]
        #         scatter[fault] = plt.scatter(latent_vectors_tsne[class_indices, 0], 
        #                                     latent_vectors_tsne[class_indices, 1], 
        #                                     label=fault, alpha=0.5, s=10)  # 점 크기 조정

        # # 범례 추가
        # plt.title("t-SNE Visualization of Latent Vectors")
        # plt.xlabel("t-SNE Component 1")
        # plt.ylabel("t-SNE Component 2")
        # plt.legend(title="Class Labels")  # 범례 제목
        # plt.savefig('./tmp_2d.png')


        # svd = TruncatedSVD(n_components=latent_vectors.shape[1])  # 최대 차원까지 SVD 수행
        # svd.fit(latent_vectors)  
        
        # # 2. 누적 설명 분산 비율 계산
        # explained_variance_ratio = np.cumsum(svd.explained_variance_ratio_)
        
        # # 3. 원하는 설명 분산(예: 95%)을 충족하는 최소 차원 찾기
        # optimal_dim = np.argmax(explained_variance_ratio >= 0.95) + 1

        # # 4. 결과 시각화 (선택적)
        # plt.plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
        # plt.axhline(y=0.95, color='r', linestyle='--', label=f'{0.95 * 100}% variance')
        # plt.axvline(x=optimal_dim, color='g', linestyle='--', label=f'Optimal dim: {optimal_dim}')
        # plt.xlabel("Number of Components")
        # plt.ylabel("Cumulative Explained Variance")
        # plt.title("SVD: Choosing Optimal Latent Dimension")
        # plt.legend()
        # plt.savefig('./svd.png')

        # cov_matrix = np.cov(latent_vectors, rowvar=False)
        # vmin, vmax = np.min(cov_matrix), np.max(cov_matrix)

        # plt.figure(figsize=(10, 8))
        # sns.heatmap(cov_matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)
        # plt.title("Latent Space Covariance Matrix")
        # plt.savefig('./cov_heatmap.png')

        # # PCA로 차원 축소
        # pca = PCA(n_components=25)
        # latent_vectors_pca = pca.fit_transform(latent_vectors)

        # # 축소된 데이터에서 공분산 계산
        # cov_matrix_pca = np.cov(latent_vectors_pca, rowvar=False)

        # plt.figure(figsize=(10, 8))
        # sns.heatmap(cov_matrix_pca, cmap="coolwarm")
        # plt.title("Covariance Matrix of Top 50 Principal Components")
        # plt.savefig('./cov_heatmap_pca_only.png')

        # cov_matrix = np.cov(latent_vectors, rowvar=False)

        # # 로그 변환
        # cov_matrix_log = np.log(np.abs(cov_matrix) + 1e-10)

        # plt.figure(figsize=(10, 8))
        # sns.heatmap(cov_matrix_log, cmap="coolwarm")
        # plt.title("Latent Space Covariance Matrix (Log-transformed)")
        # plt.savefig('./cov_heatmap_log.png')

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
                latent_vectors.append(latent_vector.cpu().numpy())


                fault_label_list.append(label.item())
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

    def save(self, root):
        os.makedirs(f'{root}/', exist_ok=True)
        torch.save(self.model.state_dict(), f'{root}/model.pt')
