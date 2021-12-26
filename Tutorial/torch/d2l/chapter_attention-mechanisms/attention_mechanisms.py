#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: scuislishuai 
@license: Apache Licence 
@file: attention_mechanisms.py 
@time: 2021/12/26
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
#################################################################
# 注意力的可视化
#################################################################
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


def show_heatmaps(metrics, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    """
    显示矩阵热图
    :param metrics:
    :param xlabel:
    :param ylabel:
    :param title:
    :param figsize:
    :param cmap:
    :return:
    """
    d2l.use_svg_display()
    num_rows, num_cols = metrics.shape[0], metrics.shape[1]
    fig, axes = d2l.plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize, sharex=True, sharey=True,
                                 squeeze=False)
    for i, (row_axes, row_metrics) in enumerate(zip(axes, metrics)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_metrics)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    d2l.plt.show()


#################################################################
# 注意力汇聚
#################################################################

def f(x):
    return 2 * torch.sin(x) + x ** 0.8


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
    d2l.plt.show()


class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super(NWKernelRegression, self).__init__(**kwargs)
        self.w = nn.Parameter(data=torch.rand(size=(1,)), requires_grad=True)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        # 数据重复keys.shape[1]次，在按照keys.shape重构，实际上queries和keys的维度变成一样的
        queries = queries.repeat_interleave(repeats=keys.shape[1]).reshape((-1, keys.shape[1]))
        self.atttention_weights = nn.functional.softmax(-((queries - keys) * self.w) ** 2 / 2, dim=1)
        return torch.bmm(self.atttention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)


def masked_softmax(X, valid_lens):
    """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
    # X:3D张量， valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # valid_len是一个向量，只有一个维度，将valid_lens的长度扩展成和X.shape[0]相同。
            # 即为X的每一行都确定一个mask的长度
            valid_lens = torch.repeat_interleave(input=valid_lens, repeats=shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
        X = d2l.sequence_mask(X=X.reshape(-1, shape[-1]), valid_len=valid_lens, value=-1e6)
        # X.reshape(shape) 维度还原
        return nn.functional.softmax(X.reshape(shape), dim=-1)


if __name__ == '__main__':
    # attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
    # show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')

    # n_train = 50  # 训练样本数
    # x_train, _ = torch.sort(torch.rand(n_train) * 5)  # 排序后的训练样本
    # y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
    # x_test = torch.arange(0, 5, 0.1)  # 测试样本
    # y_truth = f(x_test)  # 测试样本的真实输出
    # n_test = len(x_test)  # 测试样本数
    # y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    # plot_kernel_reg(y_hat)
    #
    # # X_repeat的形状:(n_test,n_train),
    # # 每⼀⾏都包含着相同的测试输⼊（例如：同样的查询）
    # X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # # x_train包含着键。 attention_weights的形状： (n_test,n_train),
    # # 每⼀⾏都包含着要在给定的每个查询的值（y_train）之间分配的注意⼒权重
    # attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
    # # y_hat的每个元素都是值的加权平均值，其中的权重是注意⼒权重
    # y_hat = torch.matmul(attention_weights, y_train)
    # plot_kernel_reg(y_hat)

    # weights = torch.ones((2, 10)) * 0.1  # weights.shape: 2 * 10
    # values = torch.arange(20.0).reshape((2, 10))
    # # weights.unsqueeze(1) 在weights.shape的第二个位置添加一个空的维度，变成2 * 1 * 10
    # # values.unsqueeze(-11) 在values.shape[-1]的最后一个位置添加一个空的维度，变成2 * 10 * 1
    # result = torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))  # 2 * 1 * 1
    # print(result.shape)

    # n_train = 50  # 训练样本数
    # x_train, _ = torch.sort(torch.rand(n_train) * 5)  # 排序后的训练样本
    # y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
    # x_test = torch.arange(0, 5, 0.1)  # 测试样本
    # y_truth = f(x_test)  # 测试样本的真实输出
    # # X_tile的形状:(n_train， n_train)，每⼀⾏都包含着相同的训练输⼊
    # X_tile = x_train.repeat((n_train, 1))
    # # Y_tile的形状:(n_train， n_train)，每⼀⾏都包含着相同的训练输出
    # Y_tile = y_train.repeat((n_train, 1))
    # # keys的形状:('n_train'， 'n_train'-1)
    # keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # # values的形状:('n_train'， 'n_train'-1)
    # values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # net = NWKernelRegression()
    # loss = nn.MSELoss(reduction='none')
    # trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
    # for epoch in range(5):
    #     trainer.zero_grad()
    #     l = loss(net(x_train, keys, values), y_train)
    #     l.sum().backward()
    #     trainer.step()
    #     print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    #     animator.add(epoch + 1, float(l.sum()))
    #
    # d2l.plt.show()
    masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
