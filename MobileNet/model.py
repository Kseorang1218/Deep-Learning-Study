import torch
import torch.nn as nn

# 원본 코드 확인용
from torchvision import models
models.mobilenet_v2
# models.mobilenet

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")
        
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(hidden_dim),
                                        nn.ReLU6()))
        layers.append(
            nn.Sequential(
                # dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),

                #pw
                nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
                )
        )

        self.conv = nn.Sequential(*layers)
        self.layers = layers


    def forward(self, x):
        if self.use_res_connect:
            print(x.size())
            print(self.conv(x).size())
            print(self.conv)
            return self.conv(x) + x
        else:
            return self.conv(x)
        

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, block=InvertedResidual, dropout=0.2, width_mult=1.0):
        super().__init__()
        
        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )
        
        input_channel = int(input_channel*width_mult)
        last_channel = int(last_channel*width_mult)

        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=input_channel, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6()
        ))

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c*width_mult)
            for i in range(n):
                if i == 0:
                    layers.append(block(in_channels=input_channel, out_channels=output_channel, stride=s, expand_ratio=t))
                else:
                    layers.append(block(in_channels=input_channel, out_channels=output_channel, stride=1, expand_ratio=t))

                input_channel = output_channel

        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=last_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(last_channel)
        ))

        self.layers = nn.Sequential(*layers)

        # self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.pool = nn.AvgPool2d(7,7)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(last_channel, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
