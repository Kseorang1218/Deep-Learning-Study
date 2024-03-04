import torch
import torch.nn as nn

class BasicBlock(nn.Module): # torch.nn.Module을 상속.
    expansion= 1                        # 상속; 어떤 클래스를 만들 때 다른 클래스의 기능을 그대로 가지고오는 것.
    def __init__(self, in_channels, out_channels, stride=1, shortcut = None):
        super(BasicBlock, self).__init__()
        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        self.shortcut = shortcut

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out

class BottleNeck(nn.Module): 
    expansion= 4
    def __init__(self, in_channels, out_channels, stride=1, shortcut = None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU()
        self.shortcut = shortcut

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5):
        super(ResNet,self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        )
        self.conv2 = self._make_layer(block, num_blocks[0], out_channels=64, stride=1) #시작은 stride 2로 시작, 이후 stride 1. stride 1 부분은 make leyer 함수 안에서 블록 만들 때 구현될것..
        self.conv3 = self._make_layer(block, num_blocks[1], out_channels=128, stride=2) 
        self.conv4 = self._make_layer(block, num_blocks[2], out_channels=256, stride=2)
        self.conv5 = self._make_layer(block, num_blocks[3], out_channels=512, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu") # mode가 이해가 안 감......
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, num_blocks, out_channels, stride=1):
        shortcut = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            shortcut = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                                     nn.BatchNorm2d(out_channels * block.expansion))

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, shortcut))

        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers) # 언패킹 연산자

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class resnet:
    def ResNet18(self):
        return ResNet(BasicBlock, [2, 2, 2, 2])

    def ResNet34(self):
        return ResNet(BasicBlock, [3, 4, 6, 3])

    def ResNet50(self):
        return ResNet(BottleNeck, [3, 4, 6, 3])

    def ResNet101(self):
        return ResNet(BottleNeck, [3, 4, 23, 3])

    def ResNet152(self):
        return ResNet(BottleNeck, [3, 8, 36, 3])
    