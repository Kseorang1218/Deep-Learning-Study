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


class AutoEncoder(nn.Module):
    def __init__(self, block, layer_size_list: List, in_channels:int = 2, input_size:int = 4096):
        super(AutoEncoder, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size

        encoder_layers = []
        prev_size = in_channels * self.input_size   # prev_size = 2*4096
        for size in layer_size_list:
            encoder_layers.append(block(prev_size, size))
            prev_size = size
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        layer_size_list.reverse()
        for size in layer_size_list[1:]:    # 첫번째 값은 제외
            decoder_layers.append(block(prev_size, size))
            prev_size = size
        decoder_layers.append(block(prev_size, in_channels * self.input_size))  # 원본 인코더 입력과 같은 크기로 맞춤
        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.encoder(x)

        latent_vector = out
        
        out = self.decoder(out)
        out = out.reshape(-1, self.in_channels, self.input_size)
    
        return out, latent_vector


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(64, 2, 4096).to(device)
    print('\nData input shape:',x.shape)

    layer_size_list = [4096, 2048, 1024, 512]
    model = AutoEncoder(LinearBlock, layer_size_list, in_channels=2, input_size=4096).to(device)
    # print('\nmodel:', model)

    x_input = x.reshape(x.size(0), -1)
    print('Model input shape:', x_input.shape)

    encoded = model.encoder(x_input)
    print('Encoded output size:', encoded.size())

    decoded = model.decoder(encoded)
    print('Decoded output size:', decoded.size())

    x_output = decoded.reshape(-1, 2, 4096)
    print('Model output shape:', x_output.shape,'\n')