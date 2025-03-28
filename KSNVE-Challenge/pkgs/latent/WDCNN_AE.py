import torch
import torch.nn as nn

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv1dBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1d = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool1d = nn.MaxPool1d(2,2)

    def forward(self, x):
        out = self.conv1d(x)
        out = self.batchnorm1d(out)
        out = self.relu(out)
        out = self.maxpool1d(out)
        return out
    
class Upsample1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super(Upsample1dBlock, self).__init__()
        self.convtranspose1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.batchnorm1d = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.convtranspose1d(x)
        out = self.batchnorm1d(out)
        out = self.relu(out)
        return out


class WDCNN_AE(nn.Module):
    def __init__(self, encoder_block, decoder_block, latent_space_size, 
                 first_kernel: int=64, in_channels:int = 2, input_size:int = 4096):
        super(WDCNN_AE, self).__init__()

        self.input_size = input_size
        self.in_channels = in_channels

        # 인코더 부분
        self.encoder_conv = nn.Sequential(
            encoder_block(in_channels=in_channels, out_channels=16, kernel_size=first_kernel, stride=16, padding=24),
            encoder_block(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            encoder_block(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            encoder_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            encoder_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
        )

        lin_input = self.get_linear_input_size()

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

        self.decoder_conv = nn.Sequential(
            decoder_block(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0, output_padding=1),
            decoder_block(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            decoder_block(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            decoder_block(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            decoder_block(in_channels=16, out_channels=in_channels, kernel_size=64, stride=32, padding=24, output_padding=16),
        )

    def get_linear_input_size(self):
        with torch.no_grad():
            # dummy 텐서를 생성하여 인코더 출력 크기 계산
            dummy = torch.rand(1, self.in_channels, self.input_size)
            dummy = self.encoder_conv(dummy)
            dummy = torch.flatten(dummy, 1)
            return dummy.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.reshape(x.size(0), self.in_channels, -1)   # [batch_size, in_channels, input_size] -> [batch_size, 1, input_size*in_channels]

        out = self.encoder_conv(out)

        out = torch.flatten(out, 1)     # [batch_size, 960]
        out = self.encoder_linear(out)  # [batch_size, latent_space_size]

        latent_vector = out

        out = self.decoder_linear(out)  # [batch_size, 960]

        out = out.view(out.shape[0], 64, -1) # [batch_size, 64, 15]
        out = self.decoder_conv(out)      

        out = out.reshape(out.size(0), self.in_channels, -1)   # [batch_size, 16, latent_space_size] -> [batch_size, in_channels, input_size]
    
        return out, latent_vector


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    in_channels = 1
    sample_size = 4096
    x = torch.randn(batch_size, in_channels, sample_size).to(device)
    print('\nData input shape:',x.shape)

    latent_space_size = 2048
    model = WDCNN_AE(Conv1dBlock, Upsample1dBlock, latent_space_size, 
                     in_channels=in_channels, input_size=sample_size).to(device)
    # print('\nmodel:', model)

    x_input = x.reshape(x.size(0), in_channels, -1)
    print('Model input shape:', x_input.shape)

    encoded = model.encoder_conv(x_input)
    encoded = torch.flatten(encoded, 1)
    encoded = model.encoder_linear(encoded)
    print('Encoded output size:', encoded.size())

    decoded = model.decoder_linear(encoded)
    print('decoded output size:', decoded.size())
    decoded = decoded.view(batch_size, 64, -1)
    decoded = model.decoder_conv(decoded)
    decoded =  decoded.reshape(decoded.size(0), in_channels, -1)
    print('Data output shape:', decoded.shape,'\n')
