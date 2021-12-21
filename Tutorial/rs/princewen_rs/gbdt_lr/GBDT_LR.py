# -*- coding:utf-8 -*-

# 程序有误，数据不正确
import lightgbm as lgb

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

# 抽取数据
print('Load data...')
train_data_path = r"E:\DataSet\DataSet\kaggle\get_started\porto-seguro-safe-driver-prediction\train.csv"
test_data_path = r"E:\DataSet\DataSet\kaggle\get_started\porto-seguro-safe-driver-prediction\test.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

NUMERIC_COLS = ["ps_reg_01", "ps_reg_02", "ps_reg_03",
                "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15"]
# print(test_data.head(10))
# print(train_data.columns)
# print(train_data.target)
print(test_data.columns)

y_train = train_data.target
# 原文有错，test.csv文件中就没有target这个属性
y_test = test_data.target
x_train = train_data[NUMERIC_COLS]
x_test = test_data[NUMERIC_COLS]

# create dataset for lingtgbm
lgb_train = lgb.Dataset(x_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

num_leaf = 64

print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict and get data on leaves, training data
y_pred = gbm.predict(x_train, pred_leaf=True)

print(np.array(y_pred).shape)
print(y_pred[:10])

print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)

for i in range(0, len(y_pred)):
    temp = np.arrange(len(y_pred[0]) * num_leaf + np.array(y_pred[i]))
    transformed_training_matrix[i][temp] += 1

y_pred = gbm.predict(x_test, pred_leaf=True)
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_testing_matrix[i][temp] += 1

lm = LogisticRegression(penalty='l2', C=0.05)
lm.fit(transformed_training_matrix, y_train)
y_pred_test = lm.predict_proba(transformed_testing_matrix)
print(y_pred_test)

NE = (-1) / len(y_pred_test) * sum(
    ((1 + y_test) / 2 * np.log(y_pred_test[:, 1]) + (1 - y_test) / 2 * np.log(1 - y_pred_test[:, 1])))
print("Normalized Cross Entropy " + str(NE))
