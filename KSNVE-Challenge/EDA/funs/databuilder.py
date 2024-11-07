# databuilder.py

import pandas as pd
import numpy as np

import os

from tqdm import tqdm

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
        


if __name__=='__main__':
    directory = '../../dataset/train'
    df = make_dataframe(directory)
    print(df)