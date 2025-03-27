import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, in_channels=4096, out_channels=2048):
        super(Encoder, self).__init__()

        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(2048, affine=False)

        self.lin2 = nn.Linear(2048, 1024, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(1024, affine=False)

        self.lin3 = nn.Linear(1024, 512, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(512, affine=False)

        self.lin4 = nn.Linear(512, 256, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(256, affine=False)


        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.batchnorm1(self.lin1(x)))
        out = self.relu(self.batchnorm2(self.lin2(out)))
        out = self.relu(self.batchnorm3(self.lin3(out)))
        out = self.relu(self.batchnorm4(self.lin4(out)))
    
        return out
    
    
class Vanilla(nn.Module):
    def __init__(self, input_size, latent_space_size, in_channels):
        super(Vanilla, self).__init__()

        prev_size = in_channels * input_size
        self.encoder = Encoder(prev_size)
        self.in_channels = in_channels
        self.input_size = input_size
        self.rep_dim = latent_space_size

        self.encoder_linear = nn.Sequential(
            nn.Linear(256, latent_space_size, bias=False),
            nn.BatchNorm1d(latent_space_size, affine=False),
            nn.ReLU(),
        )

    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = self.encoder(out)

        out = torch.flatten(out, 1)
        out = self.encoder_linear(out)

        return out


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    input_size = 4096
    channel = 2
    x = torch.randn(batch_size, channel, input_size).to(device)
    print('\nData input shape:',x.shape)

    latent_space_size = 2048
    model = Vanilla(input_size=input_size, latent_space_size=latent_space_size, in_channels=channel).to(device)
    # print('\nmodel:', model)

    x_input = x.reshape(x.size(0), -1)
    print('Model input shape:', x_input.shape)

    encoded = model.encoder(x_input)
    encoded = torch.flatten(encoded, 1)
    encoded = model.encoder_linear(encoded)
    print('Encoded output size:', encoded.size())

    out = model.forward(x_input)
    print('forward output shape', out.shape,'\n')

