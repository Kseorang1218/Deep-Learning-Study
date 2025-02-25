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

def moving_average(data, window):
    return pd.Series(data).rolling(window=window, min_periods=1).mean().to_numpy()

def data_sampling(data: np.ndarray, sample_size: int, overlap: int, window: int = 3) -> List:
    """
    주어진 sample size, overlap에 따라 데이터를 샘플링하는 함수.
    예를 들어, data가 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]이고, sample_size=4, overlap=2인 경우,
    반환되는 리스트는 다음과 같다:
    [[1, 2, 3, 4], 
    [3, 4, 5, 6], 
    [5, 6, 7, 8]]

    Parameters
    ----------
    data : np.ndarray
        샘플링하고자 하는 데이터
    sample_size : int
        데이터 샘플 크기
    overlap : int
        overlap 길이

    Returns
    -------
    List
        주어진 규칙에 따라 나뉜 세그먼트들의 리스트. 각 세그먼트는 sample_size 길이의 np.ndarray 형태를 가진다.
    """
    data_list = []
    for i in range(0, len(data)-sample_size, overlap):
        data_seg = data[i : i + sample_size]
        data_seg = moving_average(data_seg, window)
        data_list.append(data_seg)

    return data_list
        
def get_data_label_arrays(
        df: pd.DataFrame, sample_size: int, overlap: int
        )-> Tuple[np.ndarray, np.ndarray]:
    """
    데이터를 샘플링하며 (데이터, 라벨) 튜플로 반환하는 함수

    Parameters
    ----------
    df : pd.DataFrame
        샘플링하고자 하는 데이터프레임
    sample_size : int
        데이터 샘플 크기
    overlap : int
        overlap 길이

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (데이터, 라벨) 튜플
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
    print('\n', df, '\n')

    train_data, train_label = get_data_label_arrays(df, 4096, 2048)
    print('train data shape:', train_data.shape)
    print('train label shape:', train_label.shape, '\n')