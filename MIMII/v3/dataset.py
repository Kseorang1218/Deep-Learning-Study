# dataset.py
from torch.utils.data import Dataset
import torch
import librosa
import re

from processing import melspectogram

sampling_rate=16000

class MIMIIDataset(Dataset):
    def __init__(self, file_list, meta2label_dic):
        self.file_list = file_list
        self.melspectogram = melspectogram(sampling_rate=sampling_rate)
        self.meta2label_dic = meta2label_dic

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        x_wav, x_mel, label = self.transform(self.file_list[idx])
        return x_wav, x_mel, label
    
    def transform(self, filename):
        machinetype = filename.split("/")[-3]
        modelID = re.findall('id_[0-9][0-9]', filename)[0]
        label = self.meta2label_dic[f"{machinetype}-{modelID}"]
        x, _ = librosa.load(filename, sr=sampling_rate)
        x_wav = torch.from_numpy(x)
        x_mel = melspectogram(x_wav)
        return x_wav, x_mel, label

        
