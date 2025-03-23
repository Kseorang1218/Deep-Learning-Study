import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=24):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, 3, 1, 1)
        self.batchnorm2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, 3, 1, 1)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 64, 3, 1, 1)
        self.batchnorm4 = nn.BatchNorm1d(64)

        self.conv5 = nn.Conv1d(64, 64, 3, 1, 0)
        self.batchnorm5 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.maxpool1d = nn.MaxPool1d(2,2)

    def forward(self, x):
        out = self.maxpool1d(self.relu(self.batchnorm1(self.conv1(x))))
        out = self.maxpool1d(self.relu(self.batchnorm2(self.conv2(out))))
        out = self.maxpool1d(self.relu(self.batchnorm3(self.conv3(out))))
        out = self.maxpool1d(self.relu(self.batchnorm4(self.conv4(out))))
        out = self.maxpool1d(self.relu(self.batchnorm5(self.conv5(out))))
    
        return out
    
class Decoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0, output_padding=1, last_channel=1):
        super(Decoder, self).__init__()

        self.dconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.dconv2 = nn.ConvTranspose1d(64, 64, 3, 2, 1, 1)
        self.batchnorm2 = nn.BatchNorm1d(64)

        self.dconv3 = nn.ConvTranspose1d(64, 32, 3, 2, 1, 1)
        self.batchnorm3 = nn.BatchNorm1d(32)

        self.dconv4 = nn.ConvTranspose1d(32, 16, 3, 2, 1, 1)
        self.batchnorm4 = nn.BatchNorm1d(16)

        self.dconv5 = nn.ConvTranspose1d(16, last_channel, 64, 32, 24, 16)
        self.batchnorm5 = nn.BatchNorm1d(last_channel)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.batchnorm1(self.dconv1(x)))
        out = self.relu(self.batchnorm2(self.dconv2(out)))
        out = self.relu(self.batchnorm3(self.dconv3(out)))
        out = self.relu(self.batchnorm4(self.dconv4(out)))
        out = self.relu(self.batchnorm5(self.dconv5(out)))
    
        return out
    
    
class AE_Net(nn.Module):
    def __init__(self, input_size, latent_space_size, in_channels):
        super(AE_Net, self).__init__()

        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(last_channel=in_channels)
        self.in_channels = in_channels
        self.input_size = input_size

        lin_input = self.get_linear_input_size()
        # print(lin_input)

        self.encoder_linear = nn.Sequential(
            nn.Linear(lin_input, latent_space_size),
            nn.BatchNorm1d(latent_space_size),
            nn.ReLU(),
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_space_size, lin_input),
            nn.BatchNorm1d(lin_input),
            nn.ReLU()
        )

    def get_linear_input_size(self):
        with torch.no_grad():
            # dummy 텐서를 생성하여 인코더 출력 크기 계산
            dummy = torch.rand(1, self.in_channels, self.input_size)
            dummy = self.encoder(dummy)
            dummy = torch.flatten(dummy, 1)
            return dummy.shape[1]

    def forward(self, x):
        out = x.reshape(x.size(0), self.in_channels, -1)
        out = self.encoder(out)

        out = torch.flatten(out, 1)
        out = self.encoder_linear(out)

        latent_vector = out

        out = self.decoder_linear(out) 

        out = out.view(out.shape[0], 64, -1)
        out = self.decoder(out)

        out = out.reshape(out.size(0), self.in_channels, -1) 

        return out, latent_vector


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    input_size = 4096
    channel = 2
    x = torch.randn(batch_size, channel, input_size).to(device)
    print('\nData input shape:',x.shape)

    latent_space_size = 2048
    model = AE_Net(input_size, latent_space_size, channel).to(device)
    # print('\nmodel:', model)

    x_input = x.reshape(x.size(0), channel, -1)
    print('Model input shape:', x_input.shape)

    encoded = model.encoder(x_input)
    encoded = torch.flatten(encoded, 1)
    encoded = model.encoder_linear(encoded)
    print('Encoded output size:', encoded.size())

    decoded = model.decoder_linear(encoded)
    decoded = decoded.view(batch_size, 64, -1)
    decoded = model.decoder(decoded)
    decoded =  decoded.reshape(decoded.size(0), channel, -1)
    print('Data output shape:', decoded.shape,'\n')
