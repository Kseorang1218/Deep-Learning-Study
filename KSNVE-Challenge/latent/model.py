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
