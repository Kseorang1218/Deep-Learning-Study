# databuilder.py

import pandas as pd
import numpy as np

import os

from tqdm import tqdm

from typing import List, Tuple


def make_dataframe(directory: str) -> pd.DataFrame:
    """
    데이터프레임을 만드는 함수

    Parameters
    ----------
    directory : str
        데이터프레임을 생성할 디렉토리

    Returns
    -------
    pd.DataFrame
        파일 이름, 결함 종류, 각 축의 데이터, 라벨을 열으로 하는 데이터프레임
    """
    
    df = {}
    df['file_name'] = []
    df['fault_type'] = []
    df['xdata'] = []
    df['ydata'] = []
    df['label'] = []

    fault_type_list = ['normal', 'ball', 'inner', 'outer']

    label_dic = {
        'normal': 0,
        'ball'  : 1,
        'inner' : 2,
        'outer' : 3

    }

    file_list =  os.listdir(directory)
    for file in tqdm(sorted(file_list)):
        df['file_name'].append(file)
        file = os.path.join(directory, file)
        data = pd.read_csv(file)
        df['xdata'].append(np.array(data['bearingB_x']))
        df['ydata'].append(np.array(data['bearingB_y']))
        for fault_type in fault_type_list:
            if fault_type in file:
                df['fault_type'].append(fault_type)
                df['label'].append(label_dic[fault_type])
                break
        else:
            df['fault_type'].append('Unknown')
            df['label'].append(-1)

    return pd.DataFrame(df)

def data_sampling(data: pd.Series, sample_size: int, overlap: int) -> List:
    """_summary_

    Parameters
    ----------
    data : pd.Series
        _description_
    sample_size : int
        _description_
    overlap : int
        _description_

    Returns
    -------
    List
        _description_
    """
    data_list = []
    for i in range(0, len(data)-sample_size, overlap):
        data_seg = data[i : i + sample_size]
        data_list.append(data_seg)

    return data_list
        
def get_data_label_arrays(
        df: pd.DataFrame, sample_size: int, overlap: int
        )-> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    sample_size : int
        _description_
    overlap : int
        _description_

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        _description_
    """
    data = []
    labels = []

    for _, row in df.iterrows():
        xdata = data_sampling(row['xdata'], sample_size, overlap)
        ydata = data_sampling(row['ydata'], sample_size, overlap)
        label = row['label']
        
        for x_seg, y_seg in zip(xdata, ydata):
            data.append((x_seg, y_seg))
            labels.append(label)

    return np.array(data), np.array(labels)


if __name__=='__main__':
    directory = '../dataset/train'
    df = make_dataframe(directory)
    print(df,'\n')

    train_data, train_label = get_data_label_arrays(df, 4096, 2048)
    print('train data shape:', train_data.shape)
    print('train label shape:', train_label.shape)