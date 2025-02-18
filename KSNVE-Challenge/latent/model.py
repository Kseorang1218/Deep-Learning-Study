# model.py

import torch.nn as nn
import torch

from typing import List

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear(x))
        return out


class MLP(nn.Module):
    def __init__(self, block, layer_size_list: List, in_channels:int = 2, input_size:int = 4096):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size

        encoder_layers = []
        prev_size = in_channels * self.input_size   
        for size in layer_size_list:
            encoder_layers.append(block(prev_size, size))
            prev_size = size
        encoder_layers.append(nn.Linear(16, 4))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.encoder(x)
    
        return out
    
class WDCNN(nn.Module):
    def __init__(self, first_kernel: int=64, n_classes: int=10) -> None:
        super(WDCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            #Conv1
            torch.nn.Conv1d(1, 16, first_kernel, stride=16, padding=24),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            #Pool1
            torch.nn.MaxPool1d(2, 2),
            #Conv2
            torch.nn.Conv1d(16, 32, 3, stride=1, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            #Pool2
            torch.nn.MaxPool1d(2, 2),
            #Conv3
            torch.nn.Conv1d(32, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            #Pool3
            torch.nn.MaxPool1d(2, 2),
            #Conv4
            torch.nn.Conv1d(64, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            #Pool4
            torch.nn.MaxPool1d(2, 2),
            #Conv5
            torch.nn.Conv1d(64, 64, 3, stride=1, padding=0),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            #Pool5
            torch.nn.MaxPool1d(2, 2)
        )

        with torch.no_grad():
            dummy = torch.rand(1, 1, 4096*2)
            dummy = self.conv_layers(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        self.linear_layers = nn.Sequential(
            torch.nn.Linear(lin_input, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(100, n_classes)

        # self.reset_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1) 
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        x = self.head(x)
 
        return x


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(64, 2, 4096).to(device)
    print('\nData input shape:',x.shape)

    layer_size_list = [4096, 2048, 1024, 512]
    model = MLP(LinearBlock, layer_size_list, in_channels=2, input_size=4096).to(device)
    print('\nmodel:', model)

    x_input = x.reshape(x.size(0), -1)
    print('Model input shape:', x_input.shape)

    encoded = model.encoder(x_input)
    print('Encoded output size:', encoded.size())
