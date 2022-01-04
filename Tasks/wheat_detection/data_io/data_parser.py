#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: data_parser.py 
@time: 2022/01/03
@contact: scuisqianfeng@gmail.com
@site:  
@software: PyCharm 
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class WheatDataset(Dataset):
    def __init__(self, data_frame, dir, transforms):
        self.df = data_frame
        self.image_ids = data_frame['image_id'].unique()
        self.dir = dir
        self.transforms = transforms

    def __getitem__(self, item):
        image_ids = self.image_ids[item]
        image = cv2.imread(self.dir + '/' + image_ids + '.jpg', cv2.IMREAD_COLOR)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64)
        image /= 255.
        boxes = self.df[self.df['image_id'] == image_ids][['x', 'y', 'w', 'h']].values
        # x,y   w,h  -> (x1, y1) , (x2,y2)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = torch.ones((len(boxes)), dtype=torch.int)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes)), dtype=torch.int)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = torch.tensor(area, dtype=torch.float)
        target['image_id'] = torch.tensor(item, dtype=torch.int)
        target['iscrowd'] = iscrowd

        '''
        数据增强
        '''
        sample = {
            'image': image,
            'bboxes': target['boxes'],
            'labels': target['labels']
        }
        '''
        1. transforms
        2. *arg, **kwargs
        '''
        if self.transforms:
            sample = self.transforms(**sample)
            image = torch.tensor(sample['image'], dtype=torch.float)
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float)
            target['labels'] = torch.tensor(sample['labels'], dtype=torch.int64)

        return image, target

    def __len__(self):
        return len(self.image_ids)
