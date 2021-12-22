# -*- coding:utf-8 -*-

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from utils import get_file_size_mb

warnings.filterwarnings('ignore')

#######################################
# 工业蒸汽量预测是的数据量小，需要解决的问题也比较简单
# 一、数据的读取，数据集基本信息展示
#######################################

train_data_path = r'E:\DataSet\Tianchi\zhengqi\zhengqi_train.txt'
test_data_path = r'E:\DataSet\Tianchi\zhengqi\zhengqi_test.txt'
train_data = pd.read_csv(train_data_path, header='infer', sep='\t')
test_data = pd.read_csv(test_data_path, header='infer', sep='\t')

if __name__ == "__main__":
    """数据总量展示"""
    print('*' * 20 + ' 数据基本情况演示开始 ' + '*' * 20)
    print("训练集大小: {:.2f} MB".format(get_file_size_mb(train_data_path)))
    print("测试集大小: {:.2f} MB".format(get_file_size_mb(test_data_path)))
    # 训练集大小: 0.68 MB
    # 测试集大小: 0.45 MB
    # 数据量很小，模型应该不复杂

    """数据集基本情况展示"""
    # print('describe函数')
    print(train_data.describe())
    print('info函数')
    print(train_data.info())
    print('head函数')
    print(train_data.head(10))

    print('*' * 20 + ' 数据基本情况演示结束 ' + '*' * 20)

    """
    describe函数
               0     1      2       3       4   ...    34     35      36      37     38
    count    2889  2889   2889    2889    2889  ...  2889   2889    2889    2889   2889
    unique   1802  1760   1949    1821    1825  ...   420    225    1848    2010   1917
    top     0.875  0.38  0.066  -0.321  -0.049  ...  0.16  0.364  -2.608  -0.677  0.451
    freq        7     8      6       9       7  ...   450   1157      17       6      7

    [4 rows x 39 columns]
    info函数
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2889 entries, 0 to 2888
    Data columns (total 39 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   0       2889 non-null   object
     1   1       2889 non-null   object
     2   2       2889 non-null   object
     3   3       2889 non-null   object
     4   4       2889 non-null   object
     5   5       2889 non-null   object
     6   6       2889 non-null   object
     7   7       2889 non-null   object
     8   8       2889 non-null   object
     9   9       2889 non-null   object
     10  10      2889 non-null   object
     11  11      2889 non-null   object
     12  12      2889 non-null   object
     13  13      2889 non-null   object
     14  14      2889 non-null   object
     15  15      2889 non-null   object
     16  16      2889 non-null   object
     17  17      2889 non-null   object
     18  18      2889 non-null   object
     19  19      2889 non-null   object
     20  20      2889 non-null   object
     21  21      2889 non-null   object
     22  22      2889 non-null   object
     23  23      2889 non-null   object
     24  24      2889 non-null   object
     25  25      2889 non-null   object
     26  26      2889 non-null   object
     27  27      2889 non-null   object
     28  28      2889 non-null   object
     29  29      2889 non-null   object
     30  30      2889 non-null   object
     31  31      2889 non-null   object
     32  32      2889 non-null   object
     33  33      2889 non-null   object
     34  34      2889 non-null   object
     35  35      2889 non-null   object
     36  36      2889 non-null   object
     37  37      2889 non-null   object
     38  38      2889 non-null   object
    dtypes: object(39)
    memory usage: 880.4+ KB
    None

    head函数
          0      1       2      3      4   ...      34      35      36      37      38
    0     V0     V1      V2     V3     V4  ...     V34     V35     V36     V37  target
    1  0.566  0.016  -0.143  0.407  0.452  ...  -4.789  -5.101  -2.608  -3.508   0.175
    2  0.968  0.437   0.066  0.566  0.194  ...    0.16   0.364  -0.335   -0.73   0.676
    3  1.013  0.568   0.235   0.37  0.112  ...    0.16   0.364   0.765  -0.589   0.633
    4  0.733  0.368   0.283  0.165  0.599  ...  -0.065   0.364   0.333  -0.112   0.206
    5  0.684  0.638    0.26  0.209  0.337  ...  -0.215   0.364   -0.28  -0.028   0.384
    6  0.445  0.627   0.408   0.22  0.458  ...   -0.29   0.364  -0.191  -0.883    0.06
    7  0.889  0.416    0.64  0.356  0.224  ...   -0.29   0.364  -0.155  -1.318   0.415
    8  0.984  0.529   0.704  0.438  0.258  ...   -0.29   0.364     0.1  -0.899   0.609
    9  0.948   0.85   0.584  0.459  0.591  ...   -0.29   0.364   0.053  -0.553   0.981

    [10 rows x 39 columns]

    从数据集的基本情况来看，总共2889条数据，38个特征，一个target
    从describe函数的unique行可以看出，数据是连续型的；从top行可以看出，不同特征的取值范围是不同的，需要做标准化/归一化处理
    从describe函数的unique和head函数可以看出，target也是连续型的，因为项目是一个回归任务
    既然是回归任务，可能需要用到的模型就是LogisticRegression，SVR，lightgbm，xgboost等常见的模型
    """

    """数据集特征分析"""
    print('*' * 20 + ' 数据集特征分析开始 ' + '*' * 20)
    # 1  数据异常值检测 箱型图
    # 1.1 单个特征展示 V0
    print(train_data.columns)

    plt.figure(figsize=(4, 6))
    sns.boxplot(train_data['V0'], orient='v', width=0.5)
    plt.show()
    # 1.2 全部特征展示
    column_list = train_data.columns.to_list()[:38]
    plt.figure(figsize=(14, 9))
    for idx, col in enumerate(column_list):
        plt.subplot(8, 7, idx + 1)
        sns.boxplot(train_data[col], orient='v', width=0.5)
        plt.ylabel(col)

    plt.show()
    # 特征V14、19、22等特征的异常值较少

    # 2  数据分布的分析 直方图& Q-Q图 查看数据正态分布情况
    # 2.1 单个特征展示
    plt.figure(figsize=(6, 4))
    plt.subplot(1, 2, 1)
    sns.distplot(train_data['V0'], fit=stats.norm)
    plt.subplot(1, 2, 2)
    stats.probplot(train_data['V0'], plot=plt)
    plt.show()
    # 2.2 所有特征展示
    column_list = train_data.columns.to_list()[:38]
    plt.figure(figsize=(14, len(column_list)))
    i = 0
    for idx, col in enumerate(column_list):
        i += 1
        plt.subplot(10, 8, i)
        sns.distplot(train_data[col], fit=stats.norm)
        i += 1
        plt.subplot(10, 8, i)
        stats.probplot(train_data[col], plot=plt)
    plt.show()

    # V2、3、13、15、19、21 、29、37等特征的分布符合正态分布

    # 3  训练集和测试集的数据分布的分析 判断数据分布是否是同分布
    # 3.1 单特征分布
    plt.figure(figsize=(6, 4))
    sns.kdeplot(train_data['V0'], shade=True, color='red')
    sns.kdeplot(test_data['V0'], shade=True, color='blue')
    plt.show()

    # 3.2 所有特征的分布对比
    column_list = test_data.columns.to_list()
    plt.figure(figsize=(14, 9))
    for idx, col in enumerate(column_list):
        plt.subplot(8, 7, idx + 1)
        sns.kdeplot(train_data[col], color='red', shade=True)
        sns.kdeplot(test_data[col], color='blue', shade=True)
    plt.show()
    # V1/4/10/12/18/25/30/21/33/34/36等特征的分布基本相同，属于同分布

    # 4 既然已经明确了项目是回归模型，可以用线性回归图来分析特征是否是线性分布
    # 4.1 单特征线性回归图
    plt.figure(figsize=(6, 4))
    sns.regplot(x='V0', y='target', data=train_data, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                line_kws={'color': 'k'})
    plt.xlabel('V0')
    plt.show()

    # column_list = test_data.columns.to_list()
    plt.figure(figsize=(14, 9))
    for idx, col in enumerate(column_list):
        plt.subplot(8, 7, idx + 1)
        sns.regplot(x=col, y='target', data=train_data, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                    line_kws={'color': 'k'})
        plt.xlabel(col)
    plt.show()
    # 正向特征 V0/1/2/3/4/8/12/20/27/31
    # 负向特征 剩余特征
    # 无关特征 V9/14/17/22/25/26/28/29/30/32/33/34/35

    # 特征相关性分析
    plt.figure(figsize=(14, 14))
    corr_mat = train_data[[col for col in test_data.columns.to_list()]].corr(method='spearman')
    mask = np.zeros_like(corr_mat, dtype=np.bool)
    # 上三角部分遮挡起来, 只显示下半部分
    mask[np.triu_indices_from(mask)] = True
    # seaborn的发散调色板 h_pos 图的正负范围的锚定色调
    cmap = sns.diverging_palette(h_neg=220, h_pos=10, as_cmap=True)
    sns.heatmap(corr_mat, mask=mask, cmap=cmap, square=True, annot=True, fmt='.2f')
    plt.show()
    # 特征之间存在明显的线性相关性，需要降维
    print('*' * 20 + ' 数据集特征分析结束 ' + '*' * 20)
