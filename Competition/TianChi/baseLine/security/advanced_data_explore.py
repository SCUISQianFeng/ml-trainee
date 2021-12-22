# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def file_id_cnt_cut(x):
    """file_id_cnt & label 分析"""
    if x < 15000:
        return x // 1e3
    else:
        return 15


if __name__ == "__main__":
    train_path = r'E:/DataSet/Tianchi/security/security_train/security_train.csv'
    test_path = r'E:/DataSet/Tianchi/security/security_test/security_test.csv'

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_analysis = train[['file_id', 'label']].drop_duplicates(subset=['file_id', 'label'], keep='last')
    dic_ = train['file_id'].value_counts().to_dict()
    train_analysis['file_id_cnt'] = train_analysis['file_id'].map(dic_).values
    # train_analysis['file_id_cnt'].value_counts()
    sns.distplot(train_analysis['file_id_cnt'])
    print('There are {} data are below 10000'.format(
        np.sum(train_analysis['file_id_cnt'] <= 1e4) / train_analysis.shape[0]))

    train_analysis['file_id_cnt_cut'] = train_analysis['file_id_cnt'].map(file_id_cnt_cut).values
    plt.figure(figsize=[16, 20])
    plt.subplot(321)
    train_analysis[train_analysis['file_id_cnt_cut'] == 0]['label'].value_counts().sort_index().plot(kind='bar')
    plt.title('file_id_cnt_cut = 0')
    plt.xlabel('label')
    plt.ylabel('label_number')

    plt.subplot(322)
    train_analysis[train_analysis['file_id_cnt_cut'] == 1]['label'].value_counts().sort_index().plot(kind='bar')
    plt.title('file_id_cnt_cut = 1')
    plt.xlabel('label')
    plt.ylabel('label_number')

    plt.subplot(323)
    train_analysis[train_analysis['file_id_cnt_cut'] == 14]['label'].value_counts().sort_index().plot(kind='bar')
    plt.title('file_id_cnt_cut = 14')
    plt.xlabel('label')
    plt.ylabel('label_number')

    plt.subplot(324)
    train_analysis[train_analysis['file_id_cnt_cut'] == 15]['label'].value_counts().sort_index().plot(kind='bar')
    plt.title('file_id_cnt_cut = 15')
    plt.xlabel('label')
    plt.ylabel('label_number')

    plt.subplot(313)
    train_analysis['label'].value_counts().sort_index().plot(kind='bar')
    plt.title('All Data')
    plt.xlabel('label')
    plt.ylabel('label_number')
    plt.show()

    plt.figure(figsize=[16, 10])
    sns.swarmplot(x=train_analysis.iloc[:1000]['label'], y=train_analysis.iloc[:1000]['file_id_cnt'])
    plt.show()

    dic_ = train.groupby('file_id')['api'].nunique().to_dict()
    train_analysis['file_id_api_nunique'] = train_analysis['file_id'].map(dic_).values
    sns.distplot(train_analysis['file_id_api_nunique'])
    plt.show()
    # train_analysis['file_id_api_nunique'].describe()

    train_analysis.loc[train_analysis.file_id_api_nunique >= 100]['label'].value_counts().sort_index().plot(kind='bar')
    plt.title('File with api nunique >= 100')
    plt.xlabel('label')
    plt.ylabel('label_number')
    plt.show()

    plt.figure(figsize=[16, 10])
    sns.boxplot(x=train_analysis['label'], y=train_analysis['file_id_api_nunique'])
    dic_ = train.groupby('file_id')['index'].nunique().to_dict()
    train_analysis['file_id_index_nunique'] = train_analysis['file_id'].map(dic_).values
    # train_analysis['file_id_index_nunique'].describe()
    sns.distplot(train_analysis['file_id_index_nunique'])
    plt.show()

    plt.figure(figsize=[16, 8])
    plt.subplot(121)
    train_analysis.loc[train_analysis.file_id_index_nunique == 1]['label'].value_counts().sort_index().plot(kind='bar')
    plt.title('File with index nunique = 1')
    plt.xlabel('label')
    plt.ylabel('label_number')

    """file_id_index_nunique + label分析"""
    plt.subplot(122)
    train_analysis.loc[train_analysis.file_id_index_nunique == 5001]['label'].value_counts().sort_index().plot(
        kind='bar')
    plt.title('File with index nunique = 5001')
    plt.xlabel('label')
    plt.ylabel('label_number')
    plt.figure(figsize=[16, 10])
    sns.violinplot(x=train_analysis['label'], y=train_analysis['file_id_api_nunique'])
    plt.show()

    """file_id & index & max"""
    dic_ = train.groupby('file_id')['index'].max().to_dict()
    train_analysis['file_id_index_max'] = train_analysis['file_id'].map(dic_).values
    sns.distplot(train_analysis['file_id_index_max'])
    plt.figure(figsize=[16, 10])
    sns.violinplot(x=train_analysis['label'], y=train_analysis['file_id_index_max'])
    plt.figure(figsize=[16, 10])
    sns.stripplot(x=train_analysis['label'], y=train_analysis['file_id_index_max'])
    plt.show()
    """file_id & tid 分析"""
    dic_ = train.groupby('file_id')['tid'].nunique().to_dict()
    train_analysis['file_id_tid_nunique'] = train_analysis['file_id'].map(dic_).values
    # train_analysis['file_id_tid_nunique'].describe()
    sns.distplot(train_analysis['file_id_tid_nunique'])
    """file_id_tid_nunique & label 分析"""
    plt.figure(figsize=[16, 8])
    plt.subplot(121)
    train_analysis.loc[train_analysis.file_id_tid_nunique < 5]['label'].value_counts().sort_index().plot(kind='bar')
    plt.title('File with tid nunique < 5')
    plt.xlabel('label')
    plt.ylabel('label_number')

    plt.figure(figsize=[12, 8])
    sns.boxplot(x=train_analysis['label'], y=train_analysis['file_id_tid_nunique'])

    plt.figure(figsize=[12, 8])
    sns.violinplot(x=train_analysis['label'], y=train_analysis['file_id_tid_nunique'])
    plt.show()

    """file_id & tid & max"""
    dic_ = train.groupby('file_id')['tid'].max().to_dict()
    train_analysis['file_id_tid_max'] = train_analysis['file_id'].map(dic_).values

    # train_analysis['file_id_tid_max'].describe()
    plt.figure(figsize=[16, 8])
    plt.subplot(121)
    train_analysis.loc[train_analysis.file_id_tid_max >= 3000]['label'].value_counts().sort_index().plot(kind='bar')
    plt.title('File with tid max >= 3000')
    plt.xlabel('label')
    plt.ylabel('label_number')

    plt.subplot(122)
    train_analysis['label'].value_counts().sort_index().plot(kind='bar')
    plt.title('All Data')
    plt.xlabel('label')
    plt.ylabel('label_number')
    plt.show()

    """api & label"""
    train['api_label'] = train['api'] + '_' + train['label'].astype(str)
    dic_ = train['api_label'].value_counts().to_dict()
    df_api_label = pd.DataFrame.from_dict(dic_, orient='index').reset_index()
    df_api_label.columns = ['api_label', 'api_label_count']
    df_api_label['label'] = df_api_label['api_label'].apply(lambda x: int(x.split('_')[-1]))
    labels = df_api_label['label'].unique()
    for label in range(8):
        print('*' * 50, label, '*' * 50)
        print(df_api_label.loc[df_api_label.label == label].sort_values('api_label_count').iloc[-5:][
                  ['api_label', 'api_label_count']])
        print('*' * 103)


"""
************************************************** 0 **************************************************
                   api_label  api_label_count
20     CryptDecodeObjectEx_0           808724
19           RegOpenKeyExW_0           815653
11  LdrGetProcedureAddress_0          1067389
9                  NtClose_0          1150929
5         RegQueryValueExW_0          1793509
*******************************************************************************************************
************************************************** 1 **************************************************
                    api_label  api_label_count
180             RegCloseKey_1            83134
160              NtReadFile_1           101051
102  LdrGetProcedureAddress_1           199218
75                  NtClose_1           268922
72         RegQueryValueExW_1           283562
*******************************************************************************************************
************************************************** 2 **************************************************
                   api_label  api_label_count
47              NtReadFile_2           429733
34          Process32NextW_2           609066
28        RegQueryValueExW_2           704073
27  LdrGetProcedureAddress_2           711169
12                 NtClose_2          1044951
*******************************************************************************************************
************************************************** 3 **************************************************
                   api_label  api_label_count
32                 NtClose_3           614574
31             RegCloseKey_3           616165
25        RegQueryValueExW_3           749380
24  LdrGetProcedureAddress_3           762139
13           RegOpenKeyExW_3           937860
*******************************************************************************************************
************************************************** 4 **************************************************
                    api_label  api_label_count
270             RegCloseKey_4            43475
257  LdrGetProcedureAddress_4            46977
238        RegQueryValueExW_4            53934
236                 NtClose_4            54087
211           RegOpenKeyExW_4            68092
*******************************************************************************************************
************************************************** 5 **************************************************
                  api_label  api_label_count
6        GetSystemMetrics_5          1381193
3                 NtClose_5          2076013
2            GetCursorPos_5          2397779
1            Thread32Next_5          4973322
0  LdrGetProcedureAddress_5          5574419
*******************************************************************************************************
************************************************** 6 **************************************************
                    api_label  api_label_count
105           RegOpenKeyExW_6           193608
99         RegQueryValueExW_6           206940
82                  NtClose_6           254385
40   LdrGetProcedureAddress_6           503839
8          NtDelayExecution_6          1197309
*******************************************************************************************************
************************************************** 7 **************************************************
                   api_label  api_label_count
18        RegQueryValueExW_7           837933
17          Process32NextW_7           856303
14        NtDelayExecution_7           937033
10                 NtClose_7          1120847
4   LdrGetProcedureAddress_7          1839155
*******************************************************************************************************
"""
