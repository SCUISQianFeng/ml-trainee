#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: train.py
@time: 2022/01/03
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
from common.model_args import init_model_args
from models.model import ObjectHandler


def main():
    args = init_model_args()
    model = ObjectHandler(model_name=args.model_name,
                          epochs=args.epochs,
                          lr=args.lr,
                          batch_size=args.batch_size)
    model.train()


if __name__ == '__main__':
    main()
