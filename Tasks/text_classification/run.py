#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: run.py 
@time: 2021/12/12
@contact: scuisqianfeng@gmail.com
@site:  
@software: PyCharm 
"""
import argparse
import sys
import time
from importlib import import_module

import numpy as np
import torch
from train_eval import train, init_network

sys.path.append('..')

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True,
#                     help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--model', type=str, default='Transformer',
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for Char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'
    embedding = 'embedding_SougouNews.npz'
    model_name = args.model
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif

        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print('Loading data...')
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    # summary(model, [128, 32, 300])
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
