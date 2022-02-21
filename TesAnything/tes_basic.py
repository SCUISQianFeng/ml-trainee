#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: tes_basic.py
@time: 2022/01/13
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader


if __name__ == '__main__':
    ## new_tensor function test #####################
    # x = torch.tensor(data=[[1, 2], [3, 4]], dtype=torch.long)
    # print(x.data.cpu().numpy())
    # y = x.new_tensor(data=[[5, 6]], dtype=torch.long)
    # print(y.data.cpu().numpy())
    # x = x + y
    # print(x.data.cpu().numpy())

    ## Embedding module test #####################
    # num_embeddings是hidden layer中的神经元个数。num_embeddings越大， 组合（映射）能力越强，
    # embedding_dim是将一个向量经过hidden layer映射后，向量的维度
    # embedding1 = nn.Embedding(num_embeddings=10, embedding_dim=3)
    # input = torch.LongTensor([[0, 2, 0, 5]])
    # print(embedding1(input))
    """
    tensor([[[-1.1924, -0.2023, -2.0859],
         [-0.6839, -0.9385, -0.0492],
         [-1.1924, -0.2023, -2.0859],
         [-0.1658,  0.0703,  0.3619]]], grad_fn=<EmbeddingBackward>)
    """
    # embedding2 = nn.Embedding(num_embeddings=100, embedding_dim=3)
    # print(embedding2(input))
    """
    tensor([[[-0.7659, -0.8888,  1.5319],
         [ 1.5097, -0.5578, -0.1715],
         [-0.7659, -0.8888,  1.5319],
         [ 1.1171, -0.8683, -0.4013]]], grad_fn=<EmbeddingBackward>)
    """

    dataLoader = DataLoader()



