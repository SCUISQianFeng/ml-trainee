#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: froward_net.py 
@time: 2022/03/29
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

import numpy as np


class Sigmoid:
    def __int__(self):
        self.param = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))
