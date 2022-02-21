#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: data_process.py 
@time: 2022/02/07
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

# 数据采样

import numpy as np
import pandas as pd

# 从raw_sample数据集里面选择出5万用户
raw_sample = pd.read_csv(r'E:\DataSet\Tianchi\ctr\raw_sample.csv')
user_sample_id = np.random.choice(raw_sample['user'].unique(), size=10000, replace=False)
raw_sample = raw_sample[raw_sample['user'].isin(user_sample_id)]
# 采样behavior_log
users = set(user_sample_id)
reader = pd.read_csv(r'E:\DataSet\Tianchi\ctr\behavior_log.csv', chunksize=1000, iterator=True)
behavior_log = []
count = 0
for chunk in reader:
    behavior_log.append(chunk[chunk['user'].isin(users)].values)
    count += 1
    if count % 1000 == 0:
        print(count, end=',')
    if count > 20000:
        break

# 把数据拼起来，然后转成DataFrame并保存到文件
behavior_logs = np.concatenate(behavior_log)
behavior_logs = pd.DataFrame(behavior_logs, columns=['user', 'timestamp', 'btag', 'cate_id', 'brand_id'])

behavior_logs.to_csv(r'E:\DataSet\Tianchi\ctr\behavior_log_sample.csv', index=False)
raw_sample.to_csv(r'E:\DataSet\Tianchi\ctr\raw_sample_sample.csv', index=False)