import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 4)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(4, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class AutoencoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        orig_size = x.shape  # (B, C, F, T)
        
        x = torch.reshape(x, (orig_size[0], -1, orig_size[-1]))  # (B, C*F, T)
        x = x.transpose(1, 2)  # (B, T, C*F)
        x = torch.reshape(x, (-1, x.shape[-1]))  # (B*T, C*F)

        x = self.encoder(x)
        x = self.decoder(x)

        x = torch.reshape(x, (orig_size[0], orig_size[-1], -1))  # (B, T, C*F)
        x = x.transpose(1, 2)  # (B, C*F, T)
        x = torch.reshape(x, orig_size)  # (B, C, F, T)
    
        return x
    


def Autoencoder():
    return AutoencoderModel()
    n_blocks = 3
    n_channel = 128 * 2
    n_mul = 6
    frames = 51
    kernel_size = 3
    n_groups = 1
    return WaveNetModel(n_blocks, n_channel, n_mul, frames, kernel_size, n_groups)
