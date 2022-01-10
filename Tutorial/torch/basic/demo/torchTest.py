# -*- coding:utf-8 -*-

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)

# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)

if __name__ == '__main__':
    imp = torch.eye(n=5, requires_grad=True)
    one = torch.ones(size=[5,5])
    one1 = torch.ones(size=imp.shape)
    print(one)
    print(one1)
    # one_like = torch.ones_like(imp)
    # print(one_like)