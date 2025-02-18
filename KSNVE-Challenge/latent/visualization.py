# visualization.py

from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import numpy as np


def plot_tsne(root, latent_vectors, fault_label_list, latent_size, seed=42, n_components=2, except_IR=True):
    fault_label_list = np.array(fault_label_list, dtype=int)
    latent_vectors = np.array(latent_vectors)  # numpy 배열 변환
    
    fault_types = ["normal", "ball", "inner", "outer"]
    
    colors = {
        "normal": 'blue',  
        "ball": 'red',      
        "inner": 'orange',   
        "outer": cm.Greens(0.6)
    }
    
    # t-SNE 먼저 수행
    tsne = TSNE(n_components=n_components, random_state=seed, init='pca', learning_rate='auto', perplexity=50)
    latent_vectors_tsne = tsne.fit_transform(latent_vectors)
    
    # 3차원 시각화
    if n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, fault in enumerate(fault_types):
            if except_IR and i == 2:  # IR 클래스 스킵
                continue
            class_indices = np.where(fault_label_list == i)[0]
            ax.scatter(latent_vectors_tsne[class_indices, 0],
                      latent_vectors_tsne[class_indices, 1],
                      latent_vectors_tsne[class_indices, 2],
                      label=fault, alpha=0.5, s=10, color=colors[fault])
        ax.set_title(f"3D t-SNE Visualization of Latent Vectors, {latent_size}")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_zlabel("t-SNE Component 3")
        ax.legend(title="Class Labels")
    
    # 2D 시각화
    else:
        plt.figure(figsize=(10, 8))
        for i, fault in enumerate(fault_types):
            if except_IR and i == 2:  # IR 클래스 스킵
                continue
            class_indices = np.where(fault_label_list == i)[0]
            plt.scatter(latent_vectors_tsne[class_indices, 0],
                       latent_vectors_tsne[class_indices, 1],
                       label=fault, alpha=0.5, s=10, color=colors[fault])
        plt.title(f"t-SNE Visualization of Latent Vectors, {latent_size}")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(title="Class Labels")
    
    save_path = os.path.join(root, f"{n_components}D")
    fig_title = f"t-SNE_{latent_size}"
    if except_IR:
        fig_title += "_noIR"
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{fig_title}.png")
    plt.close()

def plot_umap(root, latent_vectors, latent_size, seed=42, except_IR=True):
    latent_vectors = np.array(latent_vectors)
    fault_types = ["normal", "ball", "inner", "outer"]

    u_map = umap.UMAP(random_state=seed)
    latent_vectors_umap = u_map.fit_transform(latent_vectors)

    plt.scatter(latent_vectors_umap[:, 0], latent_vectors_umap[:, 1], c=fault_types, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Digits dataset', fontsize=24)

    save_path = os.path.join(root, 'umap')
    fig_title = f"UMAP_{latent_size}"
    if except_IR:
        fig_title += "_noIR"
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{fig_title}.png")
