# dataset.py

import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader

import numpy as np

class KSNVEDataset(Dataset):
    def __init__(self, data, label): 
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index]).float()
        label = torch.tensor(self.label[index]).int()
        return data, label
    
    def __len__(self):
        return self.data.shape[0]
    
def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size, shuffle)


if __name__=='__main__':
    data = np.array(torch.randn(30000, 2, 4096))
    label = np.array(torch.randint(low=0, high=4, size=(30000,)))

    dataset = KSNVEDataset(data, label)

    data, label = dataset.__getitem__(2314)

    # print('Data example:', data)  
    # print('Label example:', label) 

    print('Data shape from dataset:', data.shape)
    print('Label shape from dataset:', label.shape)

    dataloader = get_dataloader(dataset, 64, True)
    dataloader_data, dataloader_label = next(iter(dataloader))

    # print('Data example from dataloader:', dataloader_data)  
    # print('Label example from dataloaders:', dataloader_label) 

    print(f"\nData shape from dataloader: {dataloader_data.size()}")
    print(f"Labels shape from dataloader: {dataloader_label.size()}")
