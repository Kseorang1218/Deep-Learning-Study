from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class NumpyDataset(Dataset):
    """
    np.ndarray 형식의 데이터셋

    Attributes
    ----------
    data: np.ndarray
        np.ndarray 형식의 데이터
    label: np.ndarray
        np.ndarray 형식의 레이블
    transfrom: torchvision.transforms.transforms.Compose
        Data transform하는 클래스
    target_transform: torchvision.transforms.transforms.Compose
        Label transform하는 클래스


    Methods
    ---------- 
    __len__:
        데이터 길이를 반환
    __getitem__(idx):
        데이터의 idx-th번째 인덱스의 데이터를 반환

    Examples
    ----------
    >>> train_dataset = NumpyDataset(data=train_data, label=train_label, 
                            transform=NpToTensor(), target_transform=NpToTensor())
    """
    def __init__(self, data, label, transform=None, target_transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        x = np.array(self.data[idx, :]).astype("float32")
        t = np.array(self.label[idx]).astype("int64")

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            t = self.target_transform(t)

        return x, t
    


# DataLoader parameter별 용도
# https://subinium.github.io/pytorch-dataloader/


def get_dataloader(
        dataset: NumpyDataset, batch_size: int, 
        shuffle: bool, num_workers: int = 1):
    """
    데이터셋으로부터 데이터로더를 만드는 함수

    Parameters
    ---------- 
    dataset: np.ndarray 데이터셋
        input 데이터셋
    batch_size: int
        배치 크기
    shuffle: bool
        overlapping을 사용할 때 각 샘플간 interval. sample_length=shift라면 overlapping이 없다.
    num_workers: int
        num_workers

    Returns
    ----------
    DataLoader
        파이토치 데이터로더

    Examples
    ----------
    >>> train_loader = get_dataloader(dataset=train_dataset,
                              batch_size=params["batch_size"],
                              shuffle=True)
    """

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)



