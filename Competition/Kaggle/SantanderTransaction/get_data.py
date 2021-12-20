# load data

import sys
from math import ceil

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split

sys.path.append('..')


def get_data():
    train_data_path = 'new_shiny_train.csv'
    train_data = pd.read_csv(train_data_path)
    train_y = train_data['target']
    train_x = train_data.drop(['target', 'ID_code'], axis=1, inplace=True)
    tensor_train_x = torch.tensor(data=train_x, dtype=torch.float32)
    tensor_train_y = torch.tensor(data=train_y, dtype=torch.float32)
    ds = TensorDataset(tensor_train_x, tensor_train_y)
    train_ds, val_ds = random_split(dataset=ds, lengths=[int(0.999 * len(ds)), ceil(0.001 * len(ds))])

    test_data_path = 'new_shiny_test.csv'
    test_data = pd.read_csv(filepath_or_buffer=test_data_path)
    test_ids = test_data['ID_code']
    test_x = test_data.drop(['ID_code'], axis=1)
    tensor_test_x = torch.tensor(data=test_x, dtype=torch.float32)
    tensor_test_y = torch.tensor(data=tensor_train_y, dtype=torch.float32)
    test_ds = TensorDataset(tensor_test_x, tensor_test_y)

    return train_ds, val_ds, test_ds, test_ids
