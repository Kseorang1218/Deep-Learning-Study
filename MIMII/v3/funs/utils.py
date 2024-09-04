# utils.py
import os
import glob
import itertools
import re
import numpy as np
from sklearn.metrics import f1_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import csv

import yaml
from box import Box

import torch
import numpy as np

import random

def load_yaml(config_yaml_file: str):
    """
    YAML 파일을 읽어와 Box 객체로 변환하는 함수.

    Parameters
    ----------
    config_yaml_file : str
        읽을 YAML 파일의 경로.

    Returns
    ----------
    config : Box
        YAML 파일의 내용을 포함한 Box 객체
    """
    with open(config_yaml_file) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
    return config

def set_seed(seed: int):
    """
    랜덤 시드 고정 함수.

    Parameters
    ----------
    seed : int
        고정할 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA를 사용하는 경우
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# TODO
def get_filename_list(dir_path, pattern='*', ext='*'):
    """
    find all extention files under directory
    :param dir_path: directory path
    :param ext: extention name, like wav, png...
    :param pattern: filename pattern for searching
    :return: files path list
    """
    filename_list = []
    for root, _, _ in os.walk(dir_path):
        file_path_pattern = os.path.join(root, f'{pattern}.{ext}')
        files = sorted(glob.glob(file_path_pattern))
        filename_list += files
    return filename_list

def metadata_to_label(directory):
    meta2label_dic = {}
    label2meta_dic = {}
    label = 0
    for data_dir in directory:
        machine = data_dir.split('/')[-2]
        id_list = get_machine_id_list(data_dir)
        for id_str in id_list:
            meta = machine + '-' + id_str
            meta2label_dic[meta] = label
            label2meta_dic[label] = meta
            label += 1
    return meta2label_dic, label2meta_dic

def get_machine_id_list(data_dir):
    machine_id_list = sorted(list(set(
        itertools.chain.from_iterable([re.findall('id_[0-9][0-9]', ext_id) for ext_id in get_filename_list(data_dir)])
    )))
    return machine_id_list

def create_val_file_list(target_dir,
                          id_name,
                          dir_name='val',
                          prefix_normal='normal',
                          prefix_anomaly='anomaly',
                          ext='wav'):
    normal_files_path = f'{target_dir}/{prefix_normal}_{id_name}*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))
    normal_labels = np.zeros(len(normal_files))

    anomaly_files_path = f'{target_dir}/{prefix_anomaly}_{id_name}*.{ext}'
    anomaly_files = sorted(glob.glob(anomaly_files_path))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    return files, labels


def create_train_file_list(target_dir,
                          id_name,
                          dir_name='train',
                          prefix_normal='normal',
                          prefix_anomaly='anomaly',
                          ext='wav'):
    normal_files_path = f'{target_dir}/{prefix_normal}_{id_name}*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))
    normal_labels = np.zeros(len(normal_files))

    anomaly_files_path = f'{target_dir}/{prefix_anomaly}_{id_name}*.{ext}'
    anomaly_files = sorted(glob.glob(anomaly_files_path))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    return files, labels


def save_model_state_dict(file_path, epoch=None, model=None, optimizer=None):
    import torch
    state_dict = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'model': model.state_dict() if model else None,
    }
    torch.save(state_dict, file_path)

def Visualize_ConfusionMatrix(label, prediction, title, root):
    if not os.path.isdir(root):
        os.makedirs(root)

    matrix = Get_ConfusionMatrix(label, prediction)
    disp = ConfusionMatrixDisplay(matrix)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)

    if title is None:
        title = ""
    
    plt.savefig(f"{root}/epoch{title}.png")
    plt.close(fig)  # 피규어 닫기
    

    
def Get_ConfusionMatrix(label, prediction):
    # 텐서일 경우 detach().cpu().numpy()로 변환
    label = [row.detach().cpu().numpy() if hasattr(row, 'detach') else row for row in label]
    prediction = [row.detach().cpu().numpy() if hasattr(row, 'detach') else row for row in prediction]

    # 넘파이 배열로 변환
    label = np.array(label)
    prediction = np.array(prediction)

    # 연속형 예측을 이진형으로 변환 (임계값 0.5 기준)
    prediction = (prediction >= 0.5).astype(int)

    matrix = confusion_matrix(label, prediction)
    return matrix


def Get_F1Score(label, prediction, epoch, root):
    if not os.path.isdir(root):
        os.makedirs(root)
    
    # 텐서일 경우 detach().cpu().numpy()로 변환
    label = [row.detach().cpu().numpy() if hasattr(row, 'detach') else row for row in label]
    prediction = [row.detach().cpu().numpy() if hasattr(row, 'detach') else row for row in prediction]

    # 넘파이 배열로 변환
    label = np.array(label)
    prediction = np.array(prediction)

    # 연속형 예측을 이진형으로 변환 (임계값 0.5 기준)
    prediction = (prediction >= 0.5).astype(int)

    F1Score = f1_score(label, prediction, average='macro')
    if epoch == None:
        epoch = ""
    
    with open(f"{root}/f1scores.log", "a") as f:
        f.write(f"[epoch] {epoch}\n")
        f.write(f"[F1 score] {F1Score}\n")
        f.write(f"\n")
        f.close()

    return F1Score

def save_csv(file_path, data: list):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)