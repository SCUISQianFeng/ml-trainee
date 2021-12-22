# -*- coding:utf-8 -*-


import pandas as pd
import numpy as np

# 特征工程部分

train_data_path = r'E:\DataSet\Tianchi\zhengqi\zhengqi_train.txt'
test_data_path = r'E:\DataSet\Tianchi\zhengqi\zhengqi_test.txt'
train_data = pd.read_csv(train_data_path, header='infer', sep='\t')
test_data = pd.read_csv(test_data_path, header='infer', sep='\t')


def min_max_trans(train_frame: pd.DataFrame, test_frame: pd.DataFrame):
    """
    min_max归一化数据
    :param train_frame: 训练集
    :param test_frame: 测试集
    :return: （训练集， 测试集）
    """
    from sklearn.preprocessing import MinMaxScaler
    feature_columns = [col for col in train_frame.columns.to_list() if col not in ['target']]
    scaler = MinMaxScaler()
    scaler = scaler.fit(train_frame[feature_columns])

    train_scaler = scaler.transform(train_frame[feature_columns])
    test_sacaler = scaler.transform(test_frame[feature_columns])

    train_scaler = pd.DataFrame(train_scaler)
    train_scaler.columns = feature_columns
    train_scaler['target'] = train_frame['target']
    test_scaler = pd.DataFrame(test_sacaler)
    test_scaler.columns = feature_columns

    return train_scaler, test_sacaler


def reduce_dimension(train_frame: pd.DataFrame, test_frame: pd.DataFrame, n: [int, float]):
    """
    降维
    :param train_frame: 训练集
    :param test_frame: 测试集
    :param n: 维度 整数：降维后的维度，浮点数：降维后所在比例
    :return:
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    train_pca_16 = pca.fit_transform(train_frame.iloc[:, :-1])
    test_pca_16 = pca.transform(test_frame)
    train_pca_16 = pd.DataFrame(train_pca_16)
    test_pca_16 = pd.DataFrame(test_pca_16)
    train_pca_16['target'] = train_frame['target']
    return train_pca_16, test_pca_16


def get_linear_reg(train_frame: pd.DataFrame):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    feature_columns = [col for col in train_frame.columns.to_list() if col not in ['target']]
    train_frame[feature_columns].fillna(0)
    train = train_frame[feature_columns]
    target = train_frame['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X=x_train, y=y_train)
    train_score = mean_squared_error(y_true=y_train, y_pred=regressor.predict(x_train))
    test_score = mean_squared_error(y_true=y_test, y_pred=regressor.predict(x_test))
    print("linear regressor train score: ", train_score)
    print("linear regressor test score: ", test_score)


def get_knn(train_frame: pd.DataFrame, n: int):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    feature_columns = [col for col in train_frame.columns.to_list() if col not in ['target']]
    train_frame[feature_columns].fillna(0)
    train = train_frame[feature_columns]
    target = train_frame['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=0)
    regressor = KNeighborsRegressor(n_neighbors=n)
    regressor.fit(X=x_train, y=y_train)
    train_score = mean_squared_error(y_true=y_train, y_pred=regressor.predict(x_train))
    test_score = mean_squared_error(y_true=y_test, y_pred=regressor.predict(x_test))
    print("knn train score: ", train_score)
    print("knn test score: ", test_score)


def get_random_forest(train_frame: pd.DataFrame, n: int):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    feature_columns = [col for col in train_frame.columns.to_list() if col not in ['target']]
    train_frame[feature_columns].fillna(0)
    train = train_frame[feature_columns]
    target = train_frame['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=0)
    regressor = RandomForestRegressor(n_estimators=n)
    regressor.fit(x_train, y_train)
    train_score = mean_squared_error(y_true=y_train, y_pred=regressor.predict(x_train))
    test_score = mean_squared_error(y_true=y_test, y_pred=regressor.predict(x_test))
    print("random_forest train score: ", train_score)
    print("random_forest test score: ", test_score)


def get_sgd_reg(train_frame: pd.DataFrame):
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    feature_columns = [col for col in train_frame.columns.to_list() if col not in ['target']]
    poly = PolynomialFeatures(degree=3)
    train_frame[feature_columns].fillna(0)
    train = train_frame[feature_columns]
    target = train_frame['target']
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=0)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)
    regressor = SGDRegressor(max_iter=1000, tol=1e-3)
    regressor.fit(x_train_poly, y_train)
    train_score = mean_squared_error(y_true=y_train, y_pred=regressor.predict(x_train_poly))
    test_score = mean_squared_error(y_true=y_test, y_pred=regressor.predict(x_test_poly))
    print("SGDRegressor train MSE: ", train_score)
    print("SGDRegressor test MSE: ", test_score)


def get_lightgbm(train_frame: pd.DataFrame):
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    feature_columns = [col for col in train_frame.columns if col not in ['target']]
    train = train_frame[feature_columns]
    target = train_frame['target']
    MSE_DICT = {'train_mse': [], 'test_mse': []}
    kf = KFold(n_splits=5, shuffle=True, random_state=2021)
    for epoch, (train_idx, test_idx) in enumerate(kf.split(train_frame)):
        train_data_i = train[train_idx]
        train_target_i = target[train_idx]
        test_data_i = train[test_idx]
        test_target_i = target[test_idx]

        reg = LGBMRegressor(learning_rate=0.01,
                            max_depth=-1,
                            n_estimators=5000,
                            boosting_type='gbdt',
                            random_state=2021,
                            objective='regression')
        reg.fit(X=train_data_i, y=train_target_i, eval_set=[(train_data_i, train_target_i),
                                                            (test_data_i, test_target_i)],
                eval_names=['Train', 'Test'],
                early_stopping_rounds=100,
                eval_metric=['MSE'],
                verbose=50)
        y_train_predict = reg.predict(train_data_i, num_iteration=reg.best_iteration_)
        y_test_predict = reg.predict(test_data_i, num_iteration=reg.best_iteration_)
        print('第{}折训练和预测 训练MSE 预测MSE'.format(epoch))
        train_mse = mean_squared_error(train_target_i, y_train_predict)
        test_mse = mean_squared_error(test_target_i, y_test_predict)
        print('----\n', '训练MSE\n', train_mse, '\n----')
        print('----\n', '预测MSE\n', test_mse, '\n----\n')
        MSE_DICT['train_mse'].append(train_mse)
        MSE_DICT['test_mse'].append(test_mse)
    print('----\n', '训练MSE\n', MSE_DICT['train_mse'], '\n', np.mean(MSE_DICT['train_mse']), '\n----')
    print('----\n', '预测MSE\n', MSE_DICT['test_mse'], '\n', np.mean(MSE_DICT['test_mse']), '\n----')


if __name__ == "__main__":
    train, test = min_max_trans(train_data, test_data)
    train, test = reduce_dimension(train, test, 16)
    # get_knn(train, 8)
    # get_random_forest(get_sgd_reg, 200)
    # get_sgd_reg(train)
    get_lightgbm(train)
