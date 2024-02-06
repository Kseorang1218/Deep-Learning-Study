import torch
import torch.nn as nn
import math


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels*r),
            nn.ReLU(),
            nn.Linear(in_channels*r, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.squeeze(x) 
        x = x.view(x.size(0), -1)
        x = self.excitation(x) 
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expand, stride=1, r=4):
        super().__init__()

        self.stride = stride
        self.expand = expand
        expand_channels = in_channels * self.expand
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if self.expand != 1:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.SiLU()
            ))
        layers.append(nn.Sequential(
            # dw
            nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, stride=1, 
                      padding=kernel_size//2, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.SiLU()
        ))

        self.conv1 = nn.Sequential(*layers)
        self.conv2 = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.se = SEBlock(expand_channels, r)

    def forward(self, x):
        res = x
        x_res = self.conv1(x)
        x_se = self.se(x_res)
        x = x_se * x_res
        x = self.conv2(x)
        if self.use_res_connect:
            x += res
        
        return x
    
class EfficientNet(nn.Module):
    def __init__(self, num_classes, width=1., depth=1., resolution=1., dropout=0.2):
        super().__init__()

        input_channel = int(width*32)
        last_channel = int(width*1280)
        settings = [
            #kernel size(k), strides(s), channels(c), layer nums(n)
            [3, 1, int(width*16), int(depth*1)],
            [3, 2, int(width*24), int(depth*2)],
            [5, 2, int(width*40), int(depth*2)],
            [3, 2, int(width*80), int(depth*3)],
            [5, 2, int(width*112), int(depth*3)],
            [5, 2, int(width*192), int(depth*4)],
            [3, 1, int(width*320), int(depth*1)]
        ]
        
        #self.upsample = nn.Upsample(scale_factor=resolution, mode='bilinear', align_corners=False) # ?????

        layers = []
        # 첫번째 레이어
        layers.append(nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, bias=False),
            nn.BatchNorm2d(input_channel),
        ))
        # 중간 레이어들
        for k,s,c,n in settings:
            if [k,s,c,n] == settings[0]:
                layers.append(MBConv(input_channel, out_channels=c, kernel_size=k, expand=1, stride=s))
                input_channel = c           
                continue
            for i in range(n):
                if i == 0:
                    layers.append(MBConv(input_channel, out_channels=c, kernel_size=k, expand=6, stride=s))
                else:
                    layers.append(MBConv(input_channel, out_channels=c, kernel_size=k, expand=6, stride=1))
                input_channel = c
        # 마지막 레이어
        layers.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.SiLU()
        ))
        self.layers = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        # x = self.upsample(x)
        x = self.layers(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class Model:
    def EfficientNetb0(self, num_classes):
        return EfficientNet(num_classes, 1.0, 1.0, 224, 0.2)
    
    def EfficientNetb1(self, num_classes):
        return EfficientNet(num_classes, 1.0, 1.1, 240, 0.2)
    
    def EfficientNetb2(self, num_classes):
        return EfficientNet(num_classes, 1.1, 1.2, 260, 0.3)
    
    def EfficientNetb3(self, num_classes):
        return EfficientNet(num_classes, 1.2, 1.4, 300, 0.3)
    
    def EfficientNetb4(self, num_classes):
        return EfficientNet(num_classes, 1.4, 1.8, 380, 0.4)
    
    def EfficientNetb5(self, num_classes):
        return EfficientNet(num_classes, 1.6, 2.2, 456, 0.4)
    
    def EfficientNetb6(self, num_classes):
        return EfficientNet(num_classes, 1.8, 2.6, 528, 0.5)
    
    def EfficientNetb7(self, num_classes):
        return EfficientNet(num_classes, 2.0, 3.1, 600, 0.5)
    
