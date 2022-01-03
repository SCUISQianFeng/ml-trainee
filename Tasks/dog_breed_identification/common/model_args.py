#!/usr/bin/env python  
# -*- coding:utf-8 -*-
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: model_args.py 
@time: 2022/01/02
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
import argparse


def init_model_args():
    parser = argparse.ArgumentParser(description='训练 kaggle 狗品种分类比赛')
    # training size
    parser.add_argument('--model_name', type=str, default='resnet34',
                        help='choose a model, such as resnet34')
    parser.add_argument('--data_path', type=str, default=r'E:\DataSet\DataSet\kaggle\dog-breed-identification')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--learning_rate_period', type=int, default=2)
    parser.add_argument('--learning_rate_decay', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--model_name', type=str, required=True, default='resnet34',
    #                     help='choose a model, such as resnet34')
    # todo other hyper parameters
    args = parser.parse_args()
    return args
