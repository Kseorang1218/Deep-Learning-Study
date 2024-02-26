import torch
import librosa
import numpy as np

class NpToTensor:
    """numpy array를 텐서로 변환"""
    def __call__(self, x):
        return torch.from_numpy(x)

class melspectogram:
    """mel-spectogram"""
    def __call__(self, x):
        mel_spec = librosa.feature.melspectrogram(y=x, sr=16000, n_fft=2048, hop_length=1024)
        return mel_spec

class ReSize:
    """resize"""
    def __init__(self, C, H, W):
        self.C = C
        self.H = H
        self.W = W

    def __call__(self, x):
        return np.resize(x, (self.C, self.H, self.W))
    
class ExpandDim:
    """데이터를 한 차원만큼 확장"""
    def __call__(self, x):
        return np.expand_dims(x, -1)
