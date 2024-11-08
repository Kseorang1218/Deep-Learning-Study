# painter.py

import matplotlib.pyplot as plt

import pandas as pd

from typing import Tuple

title_size = 20
label_size = 15

def draw_hist(
        df: pd.DataFrame, statistics: str, title: str, xlabel: str, ylabel: str, xlim: Tuple = None
        ) -> None:
    """
    데이터프레임으로부터 히스토그램을 그리는 함수

    Parameters
    ----------
    df : pd.DataFrame
        히스토그램을 그릴 데이터프레임
    statistics: str
        그리고자 하는 통계값
    title : str
        그림 제목
    xlabel : str
        c축 제목
    ylabel : str
        y축 제목
    xlim : Tuple, optional
        x축 범위, by default None
    """
    for fault_type in ['normal', 'ball', 'inner', 'outer']:
        sub_df = df[df['fault_type'] == fault_type]
        plt.hist(sub_df[statistics], bins=100, alpha=0.7, label=fault_type)

    plt.xlim(xlim)
    plt.title(title, fontsize=title_size)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.xticks(fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.legend(fontsize=label_size)
    plt.show()