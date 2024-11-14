# model.py

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channels):
        super(AutoEncoder, self).__init__()

        self.in_channels = in_channels
        
        self.flatten = nn.Flatten()

        self.encoder = nn.Sequential(
            nn.Linear(2*in_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2*in_channels), 
            nn.Sigmoid()  
        )

    def forward(self, x):
        out = self.flatten(x)

        out = self.encoder(out)
        out = self.decoder(out)
        out = out.reshape((-1, 2, self.in_channels))

        return out

