# model.py
import torch.nn as nn
from MobileFaceNet import MobileFaceNet
from ArcMarginProduct import ArcMarginProduct
import torch

class STgramMFN(nn.Module):
    def __init__(self, num_class, n_mels=128, win_length=1024, hop_length=512,
                 m=0.5, s=30, sub=1, use_arcface=False):
        super(STgramMFN, self).__init__()
        self.arcface = ArcMarginProduct(in_features=128, out_features=num_class,
                                        m=m, s=s, sub=sub) if use_arcface else use_arcface
        self.tgramnet = TgramNet(n_mels=n_mels, win_length=win_length, hop_length=hop_length)
        self.mobilefacenet = MobileFaceNet(num_class)

    # TODO
    def forward(self, x_wav, x_mel, label=None):
        x_wav, x_mel = x_wav.unsqueeze(1), x_mel.unsqueeze(1)
        x_t = self.tgramnet(x_wav).unsqueeze(1)
        x = torch.cat((x_mel, x_t), dim=1)
        out, feature = self.mobilefacenet(x, label)
        if self.arcface:
            out = self.arcface(feature, label)
        return out, feature
        


class TgramNet(nn.Module):
    def __init__(self, n_mels, win_length, hop_length):
        super(TgramNet, self).__init__()

        # LayerNorm 사용했으므로 bias=False
        self.large_kernel = nn.Conv1d(1, n_mels, win_length, hop_length, win_length//2, bias=False)
        self.conv_blocks = nn.Sequential(
            nn.LayerNorm(313),
            nn.LeakyReLU(),
            nn.Conv1d(n_mels, n_mels, 3, 1, 1, bias=False),

            nn.LayerNorm(313),
            nn.LeakyReLU(),
            nn.Conv1d(n_mels, n_mels, 3, 1, 1, bias=False),

            nn.LayerNorm(313),
            nn.LeakyReLU(),
            nn.Conv1d(n_mels, n_mels, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.large_kernel(x)
        out = self.conv_blocks(out)

        return out
    
if __name__ == '__main__':
    net = STgramMFN(num_class=10)
    x_wav = torch.randn((2, 16000*2))
    print(x_wav.size())
