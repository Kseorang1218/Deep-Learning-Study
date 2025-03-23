import torch.nn as nn
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepSVDD_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1):
        super(DeepSVDD_net, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, 3, 1, 1)
        self.batchnorm2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, 3, 1, 1)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 64, 3, 1, 1)
        self.batchnorm4 = nn.BatchNorm1d(64)

        self.conv5 = nn.Conv1d(64, 64, 3, 1, 1)
        self.batchnorm5 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.batchnorm1(self.conv1(x)))
        out = self.relu(self.batchnorm2(self.conv2(out)))
        out = self.relu(self.batchnorm3(self.conv3(out)))
        out = self.relu(self.batchnorm4(self.conv4(out)))
        out = self.relu(self.batchnorm5(self.conv5(out)))
    
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, 3, 1, 1)
        self.batchnorm2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, 3, 1, 1)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 64, 3, 1, 1)
        self.batchnorm4 = nn.BatchNorm1d(64)

        self.conv5 = nn.Conv1d(64, 64, 3, 1, 1)
        self.batchnorm5 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.batchnorm1(self.conv1(x)))
        out = self.relu(self.batchnorm2(self.conv2(out)))
        out = self.relu(self.batchnorm3(self.conv3(out)))
        out = self.relu(self.batchnorm4(self.conv4(out)))
        out = self.relu(self.batchnorm5(self.conv5(out)))
    
        return out
    
class Decoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, last_channel = 1):
        super(Decoder, self).__init__()

        self.dconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.dconv2 = nn.ConvTranspose1d(64, 64, 3, 1, 1)
        self.batchnorm2 = nn.BatchNorm1d(64)

        self.dconv3 = nn.ConvTranspose1d(64, 32, 3, 1, 1)
        self.batchnorm3 = nn.BatchNorm1d(32)

        self.dconv4 = nn.ConvTranspose1d(32, 16, 3, 1, 1)
        self.batchnorm4 = nn.BatchNorm1d(16)

        self.dconv5 = nn.ConvTranspose1d(16, last_channel, 3, 1, 2)
        self.batchnorm5 = nn.BatchNorm1d(last_channel)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.batchnorm1(self.dconv1(x)))
        out = self.relu(self.batchnorm2(self.dconv2(out)))
        out = self.relu(self.batchnorm3(self.dconv3(out)))
        out = self.relu(self.batchnorm4(self.dconv4(out)))
        out = self.relu(self.batchnorm5(self.dconv5(out)))
    
        return out
    
    
class AE1DCNN(nn.Module):
    def __init__(self, encoder, decoder, input_size, latent_space_size, in_channels):
        super(AE1DCNN, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.in_channels = in_channels

        with torch.no_grad():
            dummy = torch.rand(1, 1, input_size*in_channels).to(device)
            dummy = self.encoder(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]


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

    latent_space_size = 512
    encoder = Encoder(in_channels=channel)
    decoder = Decoder(last_channel=channel)
    model = AE1DCNN(encoder, decoder, input_size=input_size, latent_space_size=latent_space_size, in_channels=channel).to(device)
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

    out, latent_vector = model.forward(x_input)
    print('forward output shape', out.shape,'\n')
    print('latent vector shape', latent_vector.shape,'\n')



# for batch_idx, (data, label) in enumerate(train_loader):

#     fft = np.fft.fft(data) 
#     magnitude = np.abs(fft)
#     frequency = np.linspace(0, sr, len(magnitude))
# eequency)/2)]
#     left_magnitude = magnitude[:int(len(magnitude)/2)]

# if __name__=='__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     x = torch.randn(64, 2, 4096).to(device)
#     print('\nData input shape:',x.shape)

#     layer_size_list = [4096, 2048, 1024, 512]
#     model = MLP(LinearBlock, layer_size_list, in_channels=2, input_size=4096).to(device)
#     print('\nmodel:', model)

#     x_input = x.reshape(x.size(0), -1)
#     print('Model input shape:', x_input.shape)

#     encoded = model.encoder(x_input)
#     print('Encoded output size:', encoded.size())
