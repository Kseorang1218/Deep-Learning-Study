import pandas as pd
import numpy as np
from typing import Tuple


def split_dataframe(
        df: pd.DataFrame, train_ratio:float, val_ratio:float
        )-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    데이터프레임을 train, val, test df로 나누는 함수

    Parameters
    ---------- 
    df: pd.DataFrame
        input 데이터프레임
    train_ratio: float
        훈련 비율 ex) 0.6
    val_ratio: float
        검증 비율 ex) 0.2

    Returns
    ----------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,]
        튜플 (train_df, val_df, test_df)
        
    Examples
    ----------
    >>> train_df, val_df, test_df = databuilder.split_dataframe(df, 0.6, 0.2)
    """

    train_df = {"data":[], "label":[]}
    val_df = {"data":[], "label":[]}
    test_df = {"data":[], "label":[]}

    cum_val_ratio = train_ratio + val_ratio

    for _, row in df.iterrows():
        train_index = int(row["data"].size * train_ratio)
        val_index = int(row["data"].size * cum_val_ratio)
        train_df["data"].append(row["data"][:train_index])
        train_df["label"].append(row["label"])

        val_df["data"].append(row["data"][train_index:val_index])
        val_df["label"].append(row["label"])

        test_df["data"].append(row["data"][val_index:])
        test_df["label"].append(row["label"])

    train_df = pd.DataFrame(train_df)
    val_df = pd.DataFrame(val_df)
    test_df = pd.DataFrame(val_df)

    return train_df, val_df, test_df


def build_from_dataframe(
        df: pd.DataFrame, sample_length: int, shift: int, 
        is_onehot: bool = False
        )-> Tuple[np.ndarray, np.ndarray]:
    """
    데이터프레임으로부터 np.ndarray 타입의 (데이터, 라벨) 튜플을 만드는 함수

    Parameters
    ---------- 
    df: pd.DataFrame
        input 데이터프레임
    sample_length: int
        input 데이터 샘플의 길이
    shift: int
        overlapping을 사용할 때 각 샘플간 interval. sample_length=shift라면 overlapping이 없다.
    is_onehot: bool
        원핫인코딩 유무

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        튜플 (데이터, 라벨)
        
    Examples
    ----------
    >>> X_train, Y_train = databuilder.build_from_dataframe(train_df, 4096, 2048, False)
    """

    num_classes = df["label"].max() - df["label"].min() + 1
    num_data = df.shape[0]

    data = []
    label = []

    for i in range(num_data):
        data_segment = df.iloc[i]["data"]
        dataseg, labelseg = data_sampling(data_segment, sample_length, shift, 
                                          num_classes, df.iloc[i]["label"], is_onehot)
        data.append(dataseg)
        label.append(labelseg)

    data_array = np.concatenate(tuple(data), axis=0)
    label_array = np.concatenate(tuple(label), axis=0)

    return data_array, label_array
     


def data_sampling(
        data_segment: np.ndarray, sample_length: int, shift: int, 
        num_classes:int, classID:int, is_onehot: bool = False
        )-> Tuple[np.ndarray, np.ndarray]:
    """
    데이터 segment로부터 샘플들을 만드는 함수

    Parameters
    ---------- 
    data: np.ndarray
        input 데이터 segment
    sample_length: int
        input 데이터 샘플의 길이
    shift: int
        overlapping을 사용할 때 각 샘플간 interval. sample_length=shift라면 overlapping이 없다.
    num_classes: int
        클래스 개수
    classID
        데이터의 클래스 id
    is_onehot: bool
        원핫인코딩 유무

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        튜플 (데이터, 라벨)
        
    """

    sampled_data = np.array([
        data_segment[i: i+sample_length]
        for  i in range(0, len(data_segment)-sample_length, shift)
    ])
    if is_onehot:
        label = np.zeros((sampled_data.shape[0], num_classes))
        label[:, classID] = 1
    else:
        label = np.zeros((sampled_data.shape[0]))
        label = label + classID
    
    return sampled_data, label
