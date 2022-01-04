#!/usr/bin/env python  
# -*- coding:utf-8 -*-
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: resnet34.py
@time: 2022/01/02
@contact: scuisqianfeng@gmail.com
@site:  
@software: PyCharm 
"""
import torchvision
from torch import nn


class Model(nn.Module):

    def __init__(self, devices):
        super(Model, self).__init__()
        self.finetune_net = nn.Sequential()
        self.finetune_net.features = torchvision.models.resnet34(pretrained=True)
        # 定义一个新的输入网络，共有120个输出类别
        self.finetune_net.output_new = nn.Sequential(nn.Linear(in_features=1000, out_features=256),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=256, out_features=120))
        # 将模型参数分配给用于计算的CPU或GPU
        self.finetune_net = self.finetune_net.to(devices[0])
        # 冻结参数
        for param in self.finetune_net.features.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        return self.finetune_net(inputs)
