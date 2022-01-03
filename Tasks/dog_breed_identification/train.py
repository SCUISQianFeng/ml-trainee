#!/usr/bin/env python  
# -*- coding:utf-8 -*-
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: train.py
@time: 2022/01/02
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
from common.model_args import init_model_args
from models.model import ClassificationModelHandler


def main():
    args = init_model_args()
    model = ClassificationModelHandler(model_name=args.model_name,
                                       data_path=args.data_path,
                                       epochs=args.epochs,
                                       batch_size=args.batch_size,
                                       learning_rate=args.learning_rate,
                                       learning_rate_period=args.learning_rate_period,
                                       learning_rate_decay=args.learning_rate_decay,
                                       weight_decay=args.weight_decay,
                                       seed=args.seed)
    model.train()


if __name__ == '__main__':
    main()
