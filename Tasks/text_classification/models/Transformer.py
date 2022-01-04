#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: Transformer.py 
@time: 2021/12/18
@contact: scuisqianfeng@gmail.com
@site:  
@software: PyCharm 
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
import copy


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'Transformer'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 2000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2


'''Attention Is All You Need'''


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings=config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(num_embeddings=config.n_vocab, embedding_dim=config.embed,
                                          padding_idx=config.n_vocab)
        self.position_embedding = PositionalEncoding(embed=config.embed, pad_size=config.pad_size,
                                                     dropout=config.dropout, device=config.device)
        self.encoder = Encoder(dim_model=config.dim_model, num_head=config.num_head, hidden=config.hidden,
                               dropout=config.dropout)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(config.num_encoder)])
        self.fc1 = nn.Linear(in_features=config.pad_size * config.dim_model, out_features=config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        out = self.position_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(dim_model=dim_model, num_head=num_head, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        """
        位置编码
        :param embed: fp: 300
        :param pad_size: fp:32
        :param dropout:
        :param device:
        """
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pe = torch.tensor(
            [[(pos / (1000.0 ** (i // 2 * 2.0 / embed))) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = x + nn.Parameter(data=self.pe, requires_grad=False).to(self.device)  # 加上位置编码信息
        out = self.dropout(out)
        return out


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, scale=None):
        """
        self-attention
        :param q: [batch_size, len_Q, dim_Q]
        :param k: [batch_size, len_K, dim_K]
        :param v: [batch_size, len_V, dim_V]
        :param scale: 缩放因子 论文为根号dim_K
        :return: self-attention后的张量，以及attention张量
        """
        attention = torch.matmul(q, k.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, v)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(in_features=dim_model, out_features=num_head * self.dim_head)
        self.fc_K = nn.Linear(in_features=dim_model, out_features=num_head * self.dim_head)
        self.fc_V = nn.Linear(in_features=dim_model, out_features=num_head * self.dim_head)
        self.attention = ScaleDotProductAttention()
        self.fc = nn.Linear(in_features=num_head * self.dim_head, out_features=dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.fc_Q(x)
        k = self.fc_K(x)
        v = self.fc_V(x)
        q = q.view(batch_size, -1, self.dim_head)
        k = k.view(batch_size, -1, self.dim_head)
        v = v.view(batch_size, -1, self.dim_head)
        scale = k.size(-1) ** 0.5  # 默认缩放因子
        context = self.attention(q, k, v, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # short cut
        out = self.layer_norm(out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(in_features=dim_model, out_features=hidden)
        self.fc2 = nn.Linear(in_features=hidden, out_features=dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
