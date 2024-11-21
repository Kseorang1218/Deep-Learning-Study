# model.py

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channels:int = 2):
        super(AutoEncoder, self).__init__()

        self.in_channels = in_channels
        
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048), 
            nn.ReLU(),
            nn.Linear(2048, in_channels), 
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)

        return out

