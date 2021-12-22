# -*- coding:utf-8 -*-

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)

# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)