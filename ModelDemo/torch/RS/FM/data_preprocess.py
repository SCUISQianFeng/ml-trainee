#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: data_preprocess.py 
@time: 2022/01/14
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim
import time
import math


def load_data():
    rating_df = pd.read_csv(r'E:\DataSet\DataSet\kaggle\movieLens\ml-1m\ratings.dat',
                            sep='::',
                            header=None,
                            names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                            encoding='latin-1',
                            engine='python')
    # movie.rat数据处理
    movie_df = pd.read_csv(r'E:\DataSet\DataSet\kaggle\movieLens\ml-1m\movies.dat',
                           sep='::',
                           header=None,
                           names=['MovieID', 'Title', 'Genres'],
                           encoding='latin-1',
                           engine='python')
    movie_df['movieId_idx'] = movie_df['MovieID'].astype('category').cat.codes
    # user.rat数据处理
    user_df = pd.read_csv(r'E:\DataSet\DataSet\kaggle\movieLens\ml-1m\users.dat',
                          sep='::',
                          header=None,
                          names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                          encoding='latin-1',
                          engine='python')
    user_df['userId_index'] = user_df['UserID'].astype('category').cat.codes
    user_df['age_index'] = user_df['Age'].astype('category').cat.codes
    user_df['gender_index'] = user_df['Gender'].astype('category').cat.codes
    user_df['occupation_index'] = user_df['Occupation'].astype('category').cat.codes

    rating_df = rating_df.join(other=movie_df.set_index('MovieID'), on='MovieID')
    rating_df = rating_df.join(other=user_df.set_index('UserID'), on='UserID')
    # rating_df.to_csv(path_or_buf='ratings.csv', sep=',')
    return rating_df


def trunc_normal_(x: torch.Tensor, mean=0, std=1.):
    """ 数据做正则化处理 """
    return x.normal_().fmod(2).mul_(std).add_(mean)  # fmod :计算除法的元素余数


class FMModel(nn.Module):
    def __init__(self, n: int = 1024, k: int = 3):
        super(FMModel, self).__init__()
        self.w0 = nn.Parameter(data=torch.zeros(1))
        self.bias = nn.Embedding(num_embeddings=n, embedding_dim=1)
        self.embeddings = nn.Embedding(num_embeddings=n, embedding_dim=k)
        with torch.no_grad():
            trunc_normal_(self.embeddings.weight, std=0.01)
            trunc_normal_(self.bias.weight, std=0.01)

    def forward(self, X):
        emb = self.embeddings(X)
        pow_of_sum = emb.sum(dim=1).pow(2)
        sum_of_pow = emb.pow(2).sum(dim=1)
        pairwise = (pow_of_sum - sum_of_pow).sum(1) * 0.5
        bias = self.bias(X)
        return torch.sigmoid(self.w0 + bias + pairwise) * 5.5


def fit(iterator, model, optimizer, criterion):
    train_loss = 0
    model.train()
    for x, y in iterator:
        optimizer.zero_grad()
        y_hat = model(x.to(device))
        loss = criterion(y_hat, y.to(device))
        train_loss += loss.item() * x.shape[0]
        loss.backward()
        optimizer.step()
    return train_loss / len(iterator.dataset)


def test(iterator, model, criterion):
    train_loss = 0
    model.eval()
    for x, y in iterator:
        y_hat = model(x.to(device))
        loss = criterion(y_hat, y.to(device))
        train_loss += loss.item() * x.shape[0]
    return train_loss / len(iterator.dataset)


def train_n_epochs(model, n, optimizer, scheduler):
    criterion = nn.MSELoss().to(device)
    for epoch in range(n):
        start_time = time.time()
        train_loss = fit(train_dataloader, model, optimizer, criterion)
        valid_loss = test(valid_dataloader, model, criterion)
        scheduler.step()
        secs = int(time.time() - start_time)
        print(f'epoch {epoch}. time {secs}[s]')
        print(f'\ttrain rmse: {(math.sqrt(train_loss)):.4f}')
        print(f'\tvalidation rmse: {(math.sqrt(valid_loss)):.4f}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # import chardet
    #
    # with open(r'E:\DataSet\DataSet\kaggle\get_started\TheMovieLens\ml-1m\movies.dat', 'rb') as f:
    #     print(chardet.detect(f.read())['encoding'])
    rating_df = load_data()
    feature_columns = ['userId_index', 'movieId_idx', 'age_index', 'gender_index', 'occupation_index']
    # 每个特征的宽度（每个特征有多少不同的值）
    feature_sizes = {
        'userId_index': len(rating_df['userId_index'].unique()),
        'movieId_idx': len(rating_df['movieId_idx'].unique()),
        'age_index': len(rating_df['age_index'].unique()),
        'gender_index': len(rating_df['gender_index'].unique()),
        'occupation_index': len(rating_df['occupation_index'].unique())
    }
    # 每个特征在特征列表中开始的位置
    next_offset = 0
    feature_offsets = {}
    for k, v in feature_sizes.items():
        feature_offsets[k] = next_offset
        next_offset += v

    for column in feature_columns:
        # index特征列全部按特征宽度进行转换
        rating_df[column] = rating_df[column].apply(lambda c: c + feature_offsets[column])
    # print(rating_df[[*feature_columns, 'rating']][5])

    data_x = torch.tensor(rating_df[feature_columns].values)
    data_y = torch.tensor(rating_df['Rating'].values)
    dataset = TensorDataset(data_x, data_y)

    # split dataset
    batch_size = 1024
    train_n = int(len(dataset) * 0.9)
    valid_n = len(dataset) - train_n
    splits = [train_n, valid_n]
    assert sum(splits) == len(dataset)
    train_set, valid_set = torch.utils.data.random_split(dataset=dataset, lengths=splits)
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)

    model = FMModel(data_x.max() + 1, 120).to(device)
    wd = 1e-5
    lr = 0.001
    epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[7], gamma=0.1)
    criterion = nn.MSELoss().to(device)
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = fit(train_dataloader, model, optimizer, criterion)
        valid_loss = test(valid_dataloader, model, criterion)
        scheduler.step()
        secs = int(time.time() - start_time)
        print(f'epoch {epoch}. time: {secs}[s]')
        print(f'\ttrain rmse: {(math.sqrt(train_loss)):.4f}')
        print(f'\tvalidation rmse: {(math.sqrt(valid_loss)):.4f}')
