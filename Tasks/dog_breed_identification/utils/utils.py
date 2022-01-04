#!/usr/bin/env python  
# -*- coding:utf-8 -*-
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: utils.py 
@time: 2022/01/02
@contact: scuisqianfeng@gmail.com
@site:  
@software: PyCharm 
"""
import numpy as np
import torch
from torch import nn
import time
from datetime import timedelta


def set_seed(seed: int):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True


def init_network(model: nn.Module, method='xavier'):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if (len(param.shape)) >= 2:
                if method == 'xavier':
                    nn.init.xavier_normal_(tensor=param)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(tensor=param)
                else:
                    nn.init.normal_(tensor=param)
            elif 'bias' in name:
                nn.init.constant_(param, val=0)
            else:
                pass


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.
       Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def get_time_diff(start_time):
    """获取已使用时间"""
    end_time = time.time()
    diff = end_time - start_time
    return timedelta(seconds=int(round(diff)))
