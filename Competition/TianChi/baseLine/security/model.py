# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import os

from tqdm import tqdm_notebook
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Lambda, Embedding, Dropout, GRU, Bidirectional
from keras.layers import Conv1D, Conv2D, MaxPooling2D, GlobalAveragePooling1D, GlobalMaxPooling1D, Flatten
from keras.layers.merge import concatenate, Average, Dot, Maximum, Multiply
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K
from keras.layers import SpatialDropout1D
from keras.layers.wrappers import Bidirectional
from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings('ignore')


def TextCNN(max_len, max_cnt, embed_size, num_filters, kernel_size, conv_action, mask_zero):
    _input = Input(shape=(max_len,), dtype='int32')
    _embed = Embedding(max_cnt, embed_size, input_length=max_len, mask_zero=mask_zero)(_input)
    _embed = SpatialDropout1D(0.15)(_embed)
    warppers = []

    for _kernel_size in kernel_size:
        conv1d = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation=conv_action)(_embed)
        warppers.append(GlobalMaxPooling1D()(conv1d))

    fc = concatenate(warppers)
    fc = Dropout(0.5)(fc)
    # fc = BatchNormalization()(fc)
    fc = Dense(256, activation='relu')(fc)
    fc = Dropout(0.25)(fc)
    # fc = BatchNormalization()(fc)
    preds = Dense(8, activation='softmax')(fc)

    model = Model(inputs=_input, outputs=preds)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


class _Data_Preprocess:
    def __init__(self):
        self.int8_max = np.iinfo(np.int8).max
        self.int8_min = np.iinfo(np.int8).min

        self.int16_max = np.iinfo(np.int16).max
        self.int16_min = np.iinfo(np.int16).min

        self.int32_max = np.iinfo(np.int32).max
        self.int32_min = np.iinfo(np.int32).min

        self.int64_max = np.iinfo(np.int64).max
        self.int64_min = np.iinfo(np.int64).min

        self.float16_max = np.finfo(np.float16).max
        self.float16_min = np.finfo(np.float16).min

        self.float32_max = np.finfo(np.float32).max
        self.float32_min = np.finfo(np.float32).min

        self.float64_max = np.finfo(np.float64).max
        self.float64_min = np.finfo(np.float64).min

    def _get_type(self, min_val, max_val, types):
        if types == 'int':
            if max_val <= self.int8_max and min_val >= self.int8_min:
                return np.int8
            elif max_val <= self.int16_max <= max_val and min_val >= self.int16_min:
                return np.int16
            elif max_val <= self.int32_max and min_val >= self.int32_min:
                return np.int32
            return None

        elif types == 'float':
            if max_val <= self.float16_max and min_val >= self.float16_min:
                return np.float16
            if max_val <= self.float32_max and min_val >= self.float32_min:
                return np.float32
            if max_val <= self.float64_max and min_val >= self.float64_min:
                return np.float64
            return None

    def _memory_process(self, df):
        init_memory = df.memory_usage().sum() / 1024 ** 2 / 1024
        print('Original data occupies {} GB memory.'.format(init_memory))
        df_cols = df.columns

        for col in tqdm_notebook(df_cols):
            try:
                if 'float' in str(df[col].dtypes):
                    max_val = df[col].max()
                    min_val = df[col].min()
                    trans_types = self._get_type(min_val, max_val, 'float')
                    if trans_types is not None:
                        df[col] = df[col].astype(trans_types)
                elif 'int' in str(df[col].dtypes):
                    max_val = df[col].max()
                    min_val = df[col].min()
                    trans_types = self._get_type(min_val, max_val, 'int')
                    if trans_types is not None:
                        df[col] = df[col].astype(trans_types)
            except:
                print(' Can not do any process for column, {}.'.format(col))
        afterprocess_memory = df.memory_usage().sum() / 1024 ** 2 / 1024
        print('After processing, the data occupies {} GB memory.'.format(afterprocess_memory))
        return df


memory_process = _Data_Preprocess()


# 获取每个文件对应的字符串序列
def get_sequence(df, period_idx):
    seq_list = []
    for _id, begin in enumerate(period_idx[:-1]):
        seq_list.append(df.iloc[begin:period_idx[_id + 1]]['api_idx'].values)
    seq_list.append(df.iloc[period_idx[-1]:]['api_idx'].values)
    return seq_list


if __name__ == "__main__":
    train_path = r'E:/DataSet/Tianchi/security/security_train/security_train.csv'
    test_path = r'E:/DataSet/Tianchi/security/security_test/security_test.csv'

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # （字符串转化为数字）
    unique_api = train['api'].unique()
    api2index = {item: (i + 1) for i, item in enumerate(unique_api)}
    index2api = {(i + 1): item for i, item in enumerate(unique_api)}
    train['api_idx'] = train['api'].map(api2index)
    test['api_idx'] = test['api'].map(api2index)
    train_period_idx = train.file_id.drop_duplicates(keep='first').index.values
    test_period_idx = test.file_id.drop_duplicates(keep='first').index.values
    train_df = train[['file_id', 'label']].drop_duplicates(keep='first')
    test_df = test[['file_id']].drop_duplicates(keep='first')
    train_df['seq'] = get_sequence(train, train_period_idx)
    test_df['seq'] = get_sequence(test, test_period_idx)

    train_labels = pd.get_dummies(train_df.label).values
    train_seq = pad_sequences(train_df.seq.values, maxlen=6000)
    test_seq = pad_sequences(test_df.seq.values, maxlen=6000)

    """TextCNN训练和预测"""

    skf = KFold(n_splits=5, shuffle=True)
    max_len = 6000
    max_cnt = 295
    embed_size = 256
    num_filters = 64
    kernel_size = [2, 4, 6, 8, 10, 12, 14]
    conv_action = 'relu'
    mask_zero = False
    TRAIN = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    meta_train = np.zeros(shape=(len(train_seq), 8))
    meta_test = np.zeros(shape=(len(test_seq), 8))
    FLAG = True
    i = 0
    for tr_ind, te_ind in skf.split(train_labels):
        i += 1
        print('FOLD: '.format(i))
        print(len(te_ind), len(tr_ind))
        model_name = 'benchmark_textcnn_fold_' + str(i)
        X_train, X_train_label = train_seq[tr_ind], train_labels[tr_ind]
        X_val, X_val_label = train_seq[te_ind], train_labels[te_ind]

        model = TextCNN(max_len, max_cnt, embed_size, num_filters, kernel_size, conv_action, mask_zero)

        model_save_path = './NN/%s_%s.hdf5' % (model_name, embed_size)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=True)
        if TRAIN and FLAG:
            model.fit(X_train, X_train_label, validation_data=(X_val, X_val_label), epochs=100, batch_size=64,
                      shuffle=True, callbacks=[early_stopping, model_checkpoint])

        model.load_weights(model_save_path)
        pred_val = model.predict(X_val, batch_size=128, verbose=1)
        pred_test = model.predict(test_seq, batch_size=128, verbose=1)



        meta_train[te_ind] = pred_val
        meta_test += pred_test
        K.clear_session()
    meta_test /= 5.0

    """结果提交"""
    test_df['prob0'] = 0
    test_df['prob1'] = 0
    test_df['prob2'] = 0
    test_df['prob3'] = 0
    test_df['prob4'] = 0
    test_df['prob5'] = 0
    test_df['prob6'] = 0
    test_df['prob7'] = 0

    test_df[['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7']] = meta_test
    test_df[['file_id', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7']].to_csv(
        'nn_baseline_5fold.csv', index=None)
