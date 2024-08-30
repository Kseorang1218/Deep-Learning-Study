# databuilder.py

from typing import Tuple

import pandas as pd

import random

from funs.utils import *

def make_meta2label(df: pd.DataFrame) -> dict:
    '''
    메타데이터로부터 라벨 딕셔너리를 생성하는 함수

    Parameters
    ---------- 
    df: pd.DataFrame
        라벨을 만들기 위한 정보가 들어있는 데이터프레임
    
    Returns
    ----------
    dict
        기계 종류-모델 ID를 키로 하고 라벨을 값으로 가지는 딕셔너리
        
            
    Examples
    ----------
    >>> df = download_mimii('./data', '0_dB', 16000)
    >>> label_dic = make_meta2label(df)
    '''
    label_dic = {}
    label = 0

    modelid_list = df['model_id'].unique()
    machinetype_list = df['machine_type'].unique()

    for machinetype in machinetype_list:
        for model_id in modelid_list:
            label_dic[f'{machinetype}-{model_id}'] =  label
            label += 1

    return label_dic

def split_df(df: pd.DataFrame, val_model_id: list, test_model_id: list, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    데이터 프레임을 train, test, val 데이터 프레임으로 분할하는 함수
    '''
    set_seed(seed)
    val_df, remainder_val_df = make_df(df, val_model_id)
    test_df, remainder_test_df = make_df(df, test_model_id)

    train_df = pd.concat([remainder_val_df, remainder_test_df])

    return train_df, val_df, test_df


def make_df(df: pd.DataFrame, model_id_list: str):
    filtered_df = df[df['model_id'].isin(model_id_list)]

    num_abnormal =  sum(filtered_df['fault_type'] == 'abnormal')

    normal_indices = filtered_df[filtered_df['fault_type'] == 'normal'].index.tolist()
    selected_normal_indices = random.sample(normal_indices, min(num_abnormal, len(normal_indices)))
    selected_normal_df = filtered_df.loc[selected_normal_indices]
    
    selected_abormal_df = filtered_df[filtered_df['fault_type'] == 'abnormal']

    final_df = pd.concat([selected_normal_df, selected_abormal_df])
    remainder_df = filtered_df.drop(final_df.index, axis=0)

    return final_df, remainder_df

def build_from_dataframe(df: pd.DataFrame)-> Tuple[np.ndarray, np.ndarray]: 
    data_array = np.array(df['data'])
    label_array = np.array(df['machine_type']+'-'+df['model_id'])

    return data_array, label_array