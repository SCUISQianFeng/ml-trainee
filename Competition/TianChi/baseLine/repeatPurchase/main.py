# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import KFold
from scipy import sparse
import xgboost
import lightgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.model_selection import StratifiedKFold, KFold

folds = 5
seed = 1


def stacking_clf(clf, train_x, train_y, test_x, clf_name, kf, label_split=None):
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((folds, test_x.shape[0], 1))
    cv_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ['rf', 'ada', 'gb', 'et', 'lrj', 'knn', 'gnb']:
            clf.fit(tr_x, tr_y)
            pre = clf.predict_proba(te_x)
            train[test_index] = pre[:, 0].reshape(-1, 1)
            test_pre[i, :] = clf.predict_proba(test_x)[:, 0].reshape(-1, 1)
            cv_scores.append(log_loss(te_y, pre[:, 0].reshape(-1, 1)))
        elif clf_name in ['xgb']:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=None, missing=-1)
            params = {
                'booster': 'gbtree',
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'gamma': 1,
                'min_child_weight': 1.5,
                'max_depth': 5,
                'lambda': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'eta': 0.03,
                'tree_method': 'exact',
                'seed': 2021,
                'num_class': 2
            }
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'), (test_matrix, 'eval')]
            if test_matrix:
                model = clf.train(params,
                                  train_matrix,
                                  num_boost_round=num_round,
                                  evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds)
                pre = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)
                train[test_index] = pre[:, 0].reshape(-1, 1)
                test_pre[i, :] = model.predict(z, ntree_limit=model.best_ntree_limit)[:, 0].reshape(-1, 1)
                cv_scores.append(log_loss(te_y, pre[:, 0].reshape(-1, 1)))
        elif clf_name in ['lgb']:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            params = {
                'boosting_type': 'gbdt',
                # 'objective': 'dart',
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'learning_rate': 0.03,
                'tree_method': 'exact',
                'seed': 2021,
                'num_class': 2,
                'silent': True
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params,
                                  train_matrix,
                                  num_round,
                                  valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds)
                pre = model.predict(te_x, num_iteration=model.best_iteration)
                train[test_index] = pre[:, 0].reshape(-1, 1)
                test_pre[i, :] = model.predict(test_x,
                                               num_iteration=model.best_iteration)[:, 0].reshape(-1, 1)
                cv_scores.append(log_loss(te_y, pre[:, 0].reshape(-1, 1)))
        else:
            raise IOError("Please add new clf.")
        print('%s now score is :' % clf_name, cv_scores)
    test[:] = test_pre.mean(axis=0)
    print('%s_score_list:' % clf_name, cv_scores)
    print('%s_score_mean:' % clf_name, np.mean(cv_scores))
    return train.reshape(-1, 1), test.reshape(-1, 1)


def rf_clf(x_train, y_train, x_valid, kf, label_split=None):
    randomforest = RandomForestClassifier(n_estimators=1200,
                                          max_depth=20,
                                          n_jobs=-1,
                                          random_state=2021,
                                          max_features='auto',
                                          verbose=1)
    rf_train, rf_test = stacking_clf(randomforest,
                                     x_train,
                                     y_train,
                                     x_valid,
                                     'rf',
                                     kf,
                                     label_split=label_split)
    return rf_train, rf_test, 'rf'


def ada_clf(x_train, y_train, x_valid, kf, label_split=None):
    adaboost = AdaBoostClassifier(n_estimators=50,
                                  random_state=2021,
                                  learning_rate=0.01)
    ada_train, ada_test = stacking_clf(adaboost,
                                       x_train,
                                       y_train,
                                       x_valid,
                                       'ada',
                                       kf,
                                       label_split=label_split)
    return ada_train, ada_test, 'ada'


def gb_clf(x_train, y_train, x_valid, kf, label_split=None):
    gbdt = GradientBoostingClassifier(learning_rate=0.14,
                                      n_estimators=100,
                                      subsample=0.8,
                                      random_state=2017,
                                      max_depth=5,
                                      verbose=1)
    gbdt_train, gbdt_test = stacking_clf(gbdt,
                                         x_train,
                                         y_train,
                                         x_valid,
                                         'gb',
                                         kf,
                                         label_split=label_split)
    return gbdt_train, gbdt_test, 'gb'


def et_clf(x_train, y_train, x_valid, kf, label_split=None):
    extratree = ExtraTreesClassifier(n_estimators=1200,
                                     max_depth=35,
                                     max_features='auto',
                                     n_jobs=-1,
                                     random_state=2021,
                                     verbose=1)
    et_train, et_test = stacking_clf(extratree,
                                     x_train,
                                     y_train,
                                     x_valid,
                                     'et',
                                     kf,
                                     label_split=label_split)
    return et_train, et_test, 'et'


def xgb_clf(x_train, y_train, x_valid, kf, label_split=None):
    xgb_train, xgb_test = stacking_clf(xgboost,
                                       x_train,
                                       y_train,
                                       x_valid,
                                       'xgb',
                                       kf,
                                       label_split=label_split)
    return xgb_train, xgb_test, 'xgb'


def lgb_clf(x_train, y_train, x_valid, kf, label_split=None):
    lgb_train, lgb_test = stacking_clf(lightgbm,
                                       x_train,
                                       y_train,
                                       x_valid,
                                       'lgb',
                                       kf,
                                       label_split=label_split)
    return lgb_train, lgb_test, 'lgb'


def gnb_clf(x_train, y_train, x_valid, kf, label_split=None):
    gnb = GaussianNB()
    gnb_train, gnb_test = stacking_clf(gnb,
                                       x_train,
                                       y_train,
                                       x_valid,
                                       'gnb',
                                       kf,
                                       label_split=label_split)
    return gnb_train, gnb_test, 'gnb'


def lr_clf(x_train, y_train, x_valid, kf, label_split=None):
    logistic_regression = LogisticRegression(n_jobs=-1,
                                             random_state=2021,
                                             C=0.1,
                                             max_iter=200)
    lr_train, lr_test = stacking_clf(logistic_regression,
                                     x_train,
                                     y_train,
                                     x_valid,
                                     'lr',
                                     kf,
                                     label_split=label_split)
    return lr_train, lr_test, 'gnb'


def knn_clf(x_train, y_train, x_valid, kf, label_split=None):
    keighbors = KNeighborsClassifier(n_neighbors=200, n_jobs=-1)
    knn_train, knn_test = stacking_clf(keighbors,
                                       x_train,
                                       y_train,
                                       x_valid,
                                       'knn',
                                       kf,
                                       label_split=label_split)
    return knn_train, knn_test, 'knn'


def get_matrix(data):
    where_are_nan = np.isnan(data)
    where_are_inf = np.isinf(data)
    data[where_are_nan] = 0
    data[where_are_inf] = 0
    return data


if __name__ == "__main__":
    import tensorflow as tf

    print(tf.test.is_gpu_available())
    # all_data_test = pd.read_csv('all_data_test.csv', sep='\t')
    # features_columns = [c for c in all_data_test.columns if c not in ['label', 'prob', 'seller_path', 'cat_path',
    #                                                                   'brand_path', 'action_type_path', 'item_path',
    #                                                                   'time_stamp_path']]
    #
    # x_train = all_data_test[~all_data_test['label'].isna()][features_columns].values
    # y_train = all_data_test[~all_data_test['label'].isna()]['label'].values
    # x_valid = all_data_test[all_data_test['label'].isna()][features_columns].values
    #
    # x_train = np.float_(get_matrix(np.float_(x_train)))
    # y_train = np.int_(y_train)
    # x_valid = x_train
    # kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    # clf_list = [lgb_clf, xgb_clf]
    # # clf_list = [xgb_clf]
    # clf_list_col = ['xgb_clf', 'lgb_clf']
    # # clf_list_col = ['xgb_clf']
    # column_list = []
    # train_data_list = []
    # test_data_list = []
    # for clf in clf_list:
    #     train_data, test_data, clf_name = clf(x_train, y_train, x_valid, kf, label_split=None)
    #     train_data_list.append(train_data)
    #     test_data_list.append(test_data)
    #
    # train_stacking = np.concatenate(train_data_list, axis=1)
    # test_stacking = np.concatenate(test_data_list, axis=1)
    #
    # # 原始特征与Stacking特征合并
    # train = pd.DataFrame(np.concatenate([x_train, train_stacking], axis=1))
    # test = np.concatenate([x_valid, test_stacking], axis=1)
    #
    # df_train_all = pd.DataFrame(train)
    # df_train_all.columns = features_columns + clf_list_col
    # df_test_all = pd.DataFrame(test)
    # df_test_all.columns = features_columns + clf_list_col
    #
    # df_train_all['user_id'] = all_data_test[~all_data_test['label'].isna()]['user_id']
    # df_test_all['user_id'] = all_data_test[all_data_test['label'].isna()]['user_id']
    # df_train_all['label'] = all_data_test[~all_data_test['label'].isna()]['label']
    #
    # df_train_all.to_csv('train_all.csv', header=True, index=False)
    # df_test_all.to_csv('test_all.csv', header=True, index=False)
