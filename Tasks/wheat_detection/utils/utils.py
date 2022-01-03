#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: utils.py 
@time: 2022/01/03
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
import os
import re

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch.transforms import ToTensorV2



def get_train_tranforms():
    return A.Compose([
        A.Flip(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def collate_fn(batch):
    return tuple(zip(*batch))


def expand_bbox(x):
    r = re.findall('([0-9]+[.]?[0-9]*)', x)
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r


def get_data_frame(path):
    train_df = pd.read_csv(os.path.join(path, 'train.csv'))

    train_df['x'] = -1
    train_df['y'] = -1
    train_df['w'] = -1
    train_df['h'] = -1

    train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(expand_bbox))
    train_df['x'] = train_df['x'].astype(np.float64)
    train_df['y'] = train_df['y'].astype(np.float64)
    train_df['w'] = train_df['w'].astype(np.float64)
    train_df['h'] = train_df['h'].astype(np.float64)
    return train_df
