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


def load_data():
    rating_df = pd.read_csv(r'E:\DataSet\DataSet\kaggle\get_started\TheMovieLens\ml-1m\ratings.dat',
                            sep='::',
                            header=None,
                            names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                            encoding='latin-1',
                            engine='python')
    movie_df = pd.read_csv(r'E:\DataSet\DataSet\kaggle\get_started\TheMovieLens\ml-1m\movies.dat',
                           sep='::',
                           header=None,
                           names=['MovieID', 'Title', 'Genres'],
                           encoding='latin-1',
                           engine='python')
    movie_df['movieId_idx'] = movie_df['MovieID'].astype('category').cat.codes
    user_df = pd.read_csv(r'E:\DataSet\DataSet\kaggle\get_started\TheMovieLens\ml-1m\users.dat',
                          sep='::',
                          header=None,
                          names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                          encoding='latin-1',
                          engine='python')
    user_df['userId_idx'] = user_df['UserID'].astype('category').cat.codes
    rat_mov_df = rating_df.join(other=movie_df, on='MovieID', how='left')
    rat_mov_user_df = rat_mov_df.join(other=user_df, on='UserID', how='left')
    rat_mov_user_df.to_csv(path_or_buf='merge_result.csv', sep=',')


if __name__ == '__main__':
    # import chardet
    #
    # with open(r'E:\DataSet\DataSet\kaggle\get_started\TheMovieLens\ml-1m\movies.dat', 'rb') as f:
    #     print(chardet.detect(f.read())['encoding'])
    load_data()
