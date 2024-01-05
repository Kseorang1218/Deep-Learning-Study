import torch
import torch.nn as nn

class BasicBlock(nn.Module): # torch.nn.Module을 상속.
    expansion= 1                        # 상속; 어떤 클래스를 만들 때 다른 클래스의 기능을 그대로 가지고오는 것.
    def __init__(self, in_channels, out_channels, stride=1):
        '''
        super().__init__()
        super()로 기반 클래스(부모 클래스)를 초기화해줌으로써, 
        기반 클래스의 속성을 subclass가 받아오도록 한다. 
        (초기화를 하지 않으면, 부모 클래스의 속성을 사용할 수 없음)

        cf.) super().__init__() vs super(MyClass,self).__init__()
        좀 더 명확하게 super를 사용하기 위해서는 단순히 super().__init__()을 하는 것이 아니라 
        super(파생클래스, self).__init__() 을 해준다.이와 같이 적어주면 기능적으로 차이는 없지만, 
        파생클래스와 self를 넣어서 현재 클래스가 어떤 클래스인지 명확하게 표시해줄 수 있다.
        '''
        super(BasicBlock, self).__init__()
        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        if (stride != 1) or (in_channels != out_channels * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential() # identity mapping

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)

        return out

"""
class Account:
        num_accounts = 0
        def __init__(self, name):
                self.name = name
                Account.num_accounts += 1
        def __del__(self):
                Account.num_accounts -= 1

num_accounts처럼 클래스 내부에 선언된 변수를 클래스 변수라고 하며, self.name과 같이 self가 붙어 있는 변수를 인스턴스 변수라고 한다.
여러 인스턴스 간에 서로 공유해야 하는 값은 클래스 변수를 통해 바인딩해야 한다. 파이썬은 인스턴스의 네임스페이스에 없는 이름은 클래스의 네임스페이스에서 찾아보기 때문에 이러한 특성을 이용하면 클래스 변수가 모든 인스턴스에 공유될 수 있기 때문이다.
참고로 클래스 변수에 접근할 때 아래와 같이 클래스 이름을 사용할 수도 있습니다.
##############################
#  >>> Account.num_accounts  #
#  2                         #
#  >>>                       #
##############################

"""

class BottleNeck(nn.Module): 
    expansion= 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu3 = nn.ReLU()
        
        if (stride != 1) or (in_channels != out_channels * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential() # identity mapping

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = self.relu3(out)

        return out

####################################################################################################
# 파이썬은 언더바 하나로 시작한 이름들은 import하지 않는다.
# >>> from my_functions import *
# >>> func()
# 'datacamp'
# >>> _private_func()
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# NameError: name '_private_func' is not defined

# 위와 같은 에러를 방지하기 위해, from module import *가 아닌 모듈 자체를 import를 해보자. 
# >>> import my_functions
# >>> my_functions.func()
# 'datacamp'
# >>> my_functions._private_func()
# 7
####################################################################################################

class ResNet(nn.Module):
    ######################################
    # 가중치 초기화 아직 구현 못함!!!!!
    ######################################
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet,self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # padding은 어ㅓㄸㅎ게 정하는거지...
        )
        self.conv2 = self._make_layer(block, num_blocks[0], out_channels=64, stride=1) #시작은 stride 2로 시작, 이후 stride 1. stride 1 부분은 make leyer 함수 안에서 블록 만들 때 구현될것..
        self.conv3 = self._make_layer(block, num_blocks[1], out_channels=128, stride=2) 
        self.conv4 = self._make_layer(block, num_blocks[2], out_channels=256, stride=2)
        self.conv5 = self._make_layer(block, num_blocks[3], out_channels=512, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.fc1 = nn.Linear(512 * block.expansion, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def _make_layer(self, block, num_blocks, out_channels, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers) # 언패킹 연산자

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


class Model:
    def resnet18(self):
        return ResNet(BasicBlock, [2, 2, 2, 2])

    def resnet34(self):
        return ResNet(BasicBlock, [3, 4, 6, 3])

    def resnet50(self):
        return ResNet(BottleNeck, [3, 4, 6, 3])

    def resnet101(self):
        return ResNet(BottleNeck, [3, 4, 23, 3])

    def resnet152(self):
        return ResNet(BottleNeck, [3, 8, 36, 3])