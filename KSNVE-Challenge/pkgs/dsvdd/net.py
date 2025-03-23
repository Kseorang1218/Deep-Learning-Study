import torch.nn as nn
import torch

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
    
    
class Net(nn.Module):
    def __init__(self, input_size, latent_space_size, in_channels):
        super(Net, self).__init__()

        self.encoder = Encoder(in_channels)
        self.in_channels = in_channels
        self.input_size = input_size
        self.rep_dim = latent_space_size

        self.encoder_linear = nn.Sequential(
            nn.Linear(self.get_linear_input_size(), latent_space_size),
            nn.BatchNorm1d(latent_space_size),
            nn.ReLU(),
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

        return out


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    input_size = 4096
    channel = 2
    x = torch.randn(batch_size, channel, input_size).to(device)
    print('\nData input shape:',x.shape)

    latent_space_size = 2048
    model = Net(input_size=input_size, latent_space_size=latent_space_size, in_channels=channel).to(device)
    # print('\nmodel:', model)

    x_input = x.reshape(x.size(0), channel, -1)
    print('Model input shape:', x_input.shape)

    encoded = model.encoder(x_input)
    encoded = torch.flatten(encoded, 1)
    encoded = model.encoder_linear(encoded)
    print('Encoded output size:', encoded.size())

    out = model.forward(x_input)
    print('forward output shape', out.shape,'\n')



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
