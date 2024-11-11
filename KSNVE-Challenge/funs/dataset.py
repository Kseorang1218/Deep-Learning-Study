# dataset.py

import torch
from torch.utils.data import Dataset

class KSNVEDataset(Dataset):
    def __init__(self, data, label, transform = None): 
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        return torch.Tensor(self.data[index]), torch.Tensor(self.label[index])
    
    def __len__(self):
        return self.data.shape[0]