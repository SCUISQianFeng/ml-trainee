#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: model.py 
@time: 2022/01/03
@contact: scuisqianfeng@gmail.com
@site:  
@software: PyCharm 
"""

import os
from importlib import import_module

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data_io.data_parser import WheatDataset
from utils.utils import get_train_tranforms, collate_fn, get_data_frame


class ObjectHandler(object):
    def __init__(self, model_name=None, epochs=10, lr=1e-3, batch_size=256):
        self._model_name = model_name
        self._epochs = epochs
        self._lr = lr
        self._num_classes = 2
        self._batch_size = batch_size

    def build_model(self):
        model = import_module('model_zoo' + '.' + self._model_name)
        return model.Model()

    def train(self):
        path = r'E:\DataSet\DataSet\kaggle\global-wheat-detection'
        DIR = os.path.join(path, 'train')
        model = self.build_model().model_cate

        train_df = get_data_frame(path=path)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        # num_classes = 2  # 1 class (person) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        print('in_features', in_features)
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self._num_classes)

        dataset = WheatDataset(train_df, DIR, transforms=get_train_tranforms())
        data_loader = DataLoader(dataset, batch_size=self._batch_size, collate_fn=collate_fn)

        param = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(param, lr=0.001)

        device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        model.to(device)
        for epoch in range(self._epochs):
            for images, targets in data_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss = model(images, targets)
                loss = sum(i for i in loss.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())
