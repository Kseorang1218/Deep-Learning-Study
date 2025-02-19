# visualization.py

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix
import seaborn as sns

import numpy as np
import os


def plot_loss_curve(root, train_loss, val_loss, fig_title):
    plt.figure(figsize=(12, 4))
    epoch = np.arange(0, len(train_loss), 1)
    formatter = mticker.FormatStrFormatter('%.3f')

    plt.subplot(1, 2, 1)
    plt.plot(epoch, train_loss, label="Train loss", color='r')
    plt.title("Train loss")
    plt.xticks(np.arange(0, len(train_loss), 10))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch, val_loss, label="Val loss", color='b')
    plt.title("Val loss")
    plt.xticks(np.arange(0, len(train_loss), 10))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend()

    plt.tight_layout()
    
    save_path = os.path.join(root, "learning_curve")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{fig_title}.png")
    plt.close() 

def plot_confusion_matrix(root, y_true, y_pred, class_names, fig_title):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    save_path = os.path.join(root, "cm")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{fig_title}.png")
    plt.close() 