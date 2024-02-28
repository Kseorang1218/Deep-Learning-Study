import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from typing import NoReturn

def DrawGraph(acc_list: list, loss_list: list, title: str, mode: str, root: str) -> NoReturn:
    """
    train, test 그래프를 그리는 함수

    Parameters
    ---------- 
    acc_list: list
        acc list
    loss_list: list
        loss list
    title: str
        그래프 제목
    mode: str
        train or test
    root: str
        저장 위치

    Returns
    ----------
    NoReturn
        그래프를 root 위치에 저장
        
    Examples
    ----------
    >>> DrawGraph(train_acc_list, train_loss_list, 'train', mode='train', root="./saved/epoch100/")
        DrawGraph(test_acc_list, test_loss_list, 'test', mode='test', root="./saved/epoch100/")
    """
    mode = mode.lower()
    valid_modes = ['train', 'test']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    
    if not os.path.isdir(root):
        os.makedirs(root)

    x = np.arange(len(acc_list))

    fig, ax1 = plt.subplots()
    ax1.plot(x, acc_list, label= mode+' acc', color='blue')
    ax1.set_ylabel('acc')
    ax2 = ax1.twinx()
    ax2.plot(x, loss_list, label= mode+' loss', color='yellow')
    ax2.set_ylabel('loss')
    ax1.set_xlabel('epochs')

    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc="upper left")
    plt.savefig(f"{root}/{mode}.png")


# 학습 결과 이미지
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, root, device, dataloader, dataset, num_images=9):

    if not os.path.isdir(root):
        os.makedirs(root)

    was_training = model.training
    model.eval()
    images_so_far = 0
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if images_so_far < num_images:
                    row = images_so_far // 3
                    col = images_so_far % 3
                    axs[row, col].imshow(inputs.cpu().data[j].numpy().transpose((1, 2, 0)))
                    axs[row, col].axis('off')
                    axs[row, col].set_title('predicted: {}'.format(dataset.classes[preds[j]]))
                    plt.savefig(f"{root}/result.png")
                images_so_far += 1

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)