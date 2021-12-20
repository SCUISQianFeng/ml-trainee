# -*- coding:utf-8 -*-

import torch.nn as nn
from torchsummary import summary


class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        """
        resnet 的一个block都是第一个kernel是1，第二个kernel是3，第三个kernel是1。第三个的channels是第一的4倍
        :param in_channels:
        :param intermediate_channels:
        :param identity_downsample:
        :param stride:
        """
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=intermediate_channels)
        self.conv2 = nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=intermediate_channels)
        self.conv3 = nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels * self.expansion,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()  # 原始值
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 只有一个block是硬连接， 其他3个的都需要转换，stride=2,输出的size缩小一半  [(w-k+2p) / s] + 1
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            # 上一层输入图像的channel加大到4倍，size按stride进行缩放
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=intermediate_channels * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(intermediate_channels * 4)
            )
        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append((block(self.in_channels, intermediate_channels)))
        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


if __name__ == "__main__":
    net = ResNet101(img_channel=3, num_classes=1000)
    summary(model=net, input_size=(3, 224, 224))