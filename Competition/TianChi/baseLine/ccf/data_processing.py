# -*- coding:utf-8 -*-

"""
    数据预处理部分
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from utils import add_feature
from utils import add_label
from utils import read_data
from utils import get_predictors_df
from utils import get_target_df
from utils import get_predictors_df
from utils import normal_feature_generate
from utils import slide_feature_generate

sys.path.append(os.pardir)
import warnings

warnings.filterwarnings('ignore')

####################### 全局参数 ############
id_col_names = ['user_id', 'coupon_id', 'date_received']
target_col_name = 'label'
id_target_cols = ['user_id', 'coupon_id', 'date_received', 'label']
datapath = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/'
featurepath = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/feature'
resultpath = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/result'
tmppath = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/tmp'
scorepath = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/score'

if __name__ == "__main__":
    off_train_path = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/ccf_offline_stage1_train.csv'
    off_test_path = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/ccf_offline_stage1_test_revised.csv'
    on_train_path = r'E:/DataSet/Tianchi/o2oSeason1/O2O_data/ccf_online_stage1_train.csv'

    off_train = pd.read_csv(off_train_path, keep_default_na=True)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']

    off_test = pd.read_csv(off_test_path, keep_default_na=True)
    off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']

    on_train = pd.read_csv(on_train_path, keep_default_na=True)
    on_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']

    # 拷贝数据，以免在调试时重读文件
    dftrain = off_train.copy()
    dftest = off_test.copy()

    dftrain = add_feature(dftrain)
    dftrain = add_label(dftrain)
    dftest = add_feature(dftest)

    print('Offline 训练集满减情况', dftrain.if_fd.value_counts() / dftrain.if_fd.count())
    print('测试集满减情况', dftest.if_fd.value_counts() / dftest.if_fd.count())  # 满减分布情况相差较大

    # 生成特征文件 ##################
    # f1
    normal_feature_generate(f1)
    # sf2 带滑动串口的sf2
    slide_feature_generate(f2)
    # sf3 带滑动串口的sf3
    slide_feature_generate(f3)



    # 数据可视化 ##################
    # 箱线图
    fig = plt.figure(figsize=(4, 6))
    sns.boxplot(dftrain[(dftrain.label >= 0) & (dftrain.distance >= 0)]['distance'], orient="v", width=0.95)
    plt.show()

    # 直方图和Q-Q图
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 2, 1)
    sns.distplot(dftrain[(dftrain.label >= 0) & (dftrain.distance >= 0)]['distance'], fit=stats.norm)
    ax = plt.subplot(1, 2, 2)
    res = stats.probplot(dftrain[(dftrain.label >= 0) & (dftrain.distance >= 0)]['distance'], plot=plt)
    plt.show()

    # 分布对比
    ax = sns.kdeplot(dftrain[(dftrain.label >= 0) & (dftrain.discount_rate >= 0)]['discount_rate'],
                     color='red',
                     shade=True)
    ax = sns.kdeplot(dftrain[(dftrain.discount_rate >= 0)]['discount_rate'],
                     color='Blue',
                     shade=True)
    ax.set_xlabel('discount_rate')
    ax.set_ylabel('Frequency')
    ax = ax.legend(['train', 'test'])
    plt.show()

    # 线性关系
    fcols = 2
    frows = 1
    plt.figure(figsize=(8, 4))
    ax = plt.subplot(1, 2, 1)
    sns.regplot(x='distance',
                y='label',
                data=dftrain[(dftrain.label >= 0) & (dftrain.distance >= 0)][['distance', 'label']],
                ax=ax,
                scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                line_kws={'color': 'k'})
    plt.xlabel('distance')
    plt.ylabel('label')
    ax = plt.subplot(1, 2, 2)
    sns.displot(dftrain[(dftrain.label >= 0) & (dftrain.distance >= 0)]['distance'].dropna())
    plt.xlabel('distance')
    plt.show()

    # 特征总览
    traindf, testdf = read_data('sf3')
    train_X = get_predictors_df(traindf)
    train_y = get_target_df(traindf)
    test_X = get_predictors_df(testdf)

    # print(traindf.describe())
    # print(testdf.describe())

    # 画箱式图
    column = train_X.columns.tolist()[:46]
    fig = plt.figure(figsize=(20, 40))
    for i in range(45):
        plt.subplot(15, 3, i + 1)
        sns.boxplot(train_X[column[i]], orient='v', width=0.5)
        plt.ylabel(column[i], fontsize=8)
    plt.show()

    column = test_X.columns.tolist()[:46]
    fig = plt.figure(figsize=(20, 40))
    for i in range(45):
        plt.subplot(15, 3, i + 1)
        sns.boxplot(test_X[column[i]], orient='v', width=0.5)
        plt.ylabel(column[i], fontsize=8)
    plt.show()

    # 对比分布
    dist_cols = 4
    dist_rows = len(test_X.columns)

    plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

    for i, col in enumerate(test_X.columns):
        ax = plt.subplot(dist_cols, dist_rows, i + 1)
        ax = sns.kdeplot(train_X[col], color='Red', shade=True)
        ax = sns.kdeplot(test_X[col], color='Blue', shade=True)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax = ax.legend(['train', 'test'])
    plt.show()
    # 训练集中满减和不满减的数量基本相同，而测试集全是满减的数据，导致分布差别较大

    train_X_fd1 = train_X[train_X.if_fd == 1].reset_index(drop=True)
    test_X_fd1 = test_X[test_X.if_fd == 1].reset_index(drop=True)

    # 满减情况对比分布
    dist_cols = 4
    dist_rows = len(test_X_fd1.columns)
    plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

    for i, col in enumerate(test_X_fd1.columns):
        ax = plt.subplot(dist_cols, dist_rows, i + 1)
        ax = sns.kdeplot(train_X_fd1[col], color='Red', shade=True)
        ax = sns.kdeplot(test_X_fd1[col], color='Blue', shade=True)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax = ax.legend(['train', 'test'])
    plt.show()

    # 特征相关性发分析
    plt.figure(figsize=(20, 16))
    column = traindf.columns.tolist()
    mcorr = traindf[column].corr(method='spearman')
    mask = np.zeros_like(mcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
    plt.show()
