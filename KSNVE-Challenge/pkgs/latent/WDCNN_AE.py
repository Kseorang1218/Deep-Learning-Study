import torch
import torch.nn as nn

class WDCNN_AE(nn.Module):
    def __init__(self, first_kernel: int=64) -> None:
        super(WDCNN_AE, self).__init__()

        # 인코더 부분
        self.conv_layers = nn.Sequential(
            # Conv1
            torch.nn.Conv1d(1, 16, first_kernel, stride=16, padding=24),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            # Pool1
            torch.nn.MaxPool1d(2, 2),
            # Conv2
            torch.nn.Conv1d(16, 32, 3, stride=1, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            # Pool2
            torch.nn.MaxPool1d(2, 2),
            # Conv3
            torch.nn.Conv1d(32, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            # Pool3
            torch.nn.MaxPool1d(2, 2),
            # Conv4
            torch.nn.Conv1d(64, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            # Pool4
            torch.nn.MaxPool1d(2, 2),
            # Conv5
            torch.nn.Conv1d(64, 64, 3, stride=1, padding=0),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            # Pool5
            torch.nn.MaxPool1d(2, 2)
        )

        # Flatten 후 입력 크기 계산
        with torch.no_grad():
            dummy = torch.rand(1, 1, 4096*2)
            dummy = self.conv_layers(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        # 인코더의 linear layer
        self.encoder_linear = nn.Sequential(
            torch.nn.Linear(lin_input, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
        )

        # 디코더 부분
        self.decoder_linear = nn.Sequential(
            torch.nn.Linear(100, lin_input),
            torch.nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            # ConvTranspose5
            torch.nn.ConvTranspose1d(64, 64, 3, stride=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            # Conv4
            torch.nn.ConvTranspose1d(64, 64, 3, stride=2, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            # Conv3
            torch.nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            # Conv2
            torch.nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            # Conv1
            torch.nn.ConvTranspose1d(16, 1, first_kernel, stride=16, padding=24),
            torch.nn.BatchNorm1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 인코더: 입력을 압축
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.encoder_linear(x)

        latent_vector = x

        # 디코더: 압축된 정보를 복원
        x_reconstructed = self.decoder_linear(x)
        x_reconstructed = x_reconstructed.view(x_reconstructed.size(0), 64, -1)  # 디코더 입력 형태로 변형
        x_reconstructed = self.decoder_conv(x_reconstructed)

        return x_reconstructed, latent_vector
