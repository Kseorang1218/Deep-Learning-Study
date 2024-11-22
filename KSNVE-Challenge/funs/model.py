# model.py

import torch.nn as nn

import torch

class AutoEncoder(nn.Module):
    def __init__(self, in_channels:int = 2, input_size:int = 4096):
        super(AutoEncoder, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size
        
        self.encoder = nn.Sequential(
            nn.Linear(in_channels * input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048), 
            nn.ReLU(),
            nn.Linear(2048, 4096), 
            nn.ReLU(),
            nn.Linear(4096, in_channels * input_size), 
        )


    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.encoder(x)
        out = self.decoder(out)
        out = out.reshape(-1, self.in_channels, self.input_size)
        return out


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(64, 2, 4096).to(device)
    print('\nData input shape:',x.shape)
    model = AutoEncoder(in_channels=2, input_size=4096).to(device)
    # print('\nmodel:', model)

    x_input = x.reshape(x.size(0), -1)
    print('Model input shape:', x_input.shape)

    encoded = model.encoder(x_input)
    print('Encoded output size:', encoded.size())

    decoded = model.decoder(encoded)
    print('Decoded output size:', decoded.size())

    x_output = decoded.reshape(-1, 2, 4096)
    print('Model output shape:', x_output.shape,'\n')