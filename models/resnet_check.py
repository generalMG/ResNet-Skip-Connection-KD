import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #  no MaxPooling layer

        self.conv2 = self.make_block(block, 64, num_block[0], 1)
        self.conv3 = self.make_block(block, 128, num_block[1], 2)
        self.conv4 = self.make_block(block, 256, num_block[2], 2)
        self.conv5 = self.make_block(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_block(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output_1 = self.conv1(x)
        output_2 = self.conv2(output_1)
        output_3 = self.conv3(output_2)
        output_4 = self.conv4(output_3)
        output_5 = self.conv5(output_4)
        output_avg = self.avg_pool(output_5)
        output = output_avg.view(output_avg.size(0), -1)  # flattening
        output_fc = self.fc(output)

        return output_fc

class Basic(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()


        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * Basic.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * Basic.expansion))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != Basic.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Basic.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Basic.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) - self.shortcut(x))

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * Bottleneck.expansion))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

def resnet18():
    return ResNet(Basic, [2, 2, 2, 2])

def resnet34():
    return ResNet(Basic, [3, 4, 6, 3])

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])