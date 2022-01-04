#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: FastRCNN.py 
@time: 2022/01/03
@contact: scuisqianfeng@gmail.com
@site:  
@software: PyCharm 
"""
import torchvision
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # load a model pre-trained pre-trained on COCO
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    def forward(self, x):
        return self.model(x)
