# dataset.py
from torch.utils.data import Dataset

import numpy as np
import torch
import librosa
import re

from funs.processing import melspectogram

sampling_rate=16000

class MIMIIDataset(Dataset):
    def __init__(self, data, label, transform=None, target_transform = None):
        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = np.array(self.data).astype('float32')
        label = np.array(self.label).astype('int64')

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return data, label
    

    # def transform(self, filename):
    #     machinetype = filename.split("/")[-3]
    #     modelID = re.findall('id_[0-9][0-9]', filename)[0]
    #     label = self.meta2label_dic[f"{machinetype}-{modelID}"]
    #     x, _ = librosa.load(filename, sr=sampling_rate)
    #     x_wav = torch.from_numpy(x)
    #     x_mel = self.melspectogram(x_wav)
    #     return x_wav, x_mel, label

        
