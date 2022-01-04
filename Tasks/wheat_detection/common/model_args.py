#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: model_args.py 
@time: 2022/01/03
@contact: scuisqianfeng@gmail.com
@site:  
@software: PyCharm 
"""
import argparse


def init_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='FastRCNN', help='choose a model, such as FastRCNN')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    return args
