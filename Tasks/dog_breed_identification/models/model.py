#!/usr/bin/env python  
# -*- coding:utf-8 -*-
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: model.py 
@time: 2022/01/02
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
import torch
import torchvision
import os
from importlib import import_module
from utils.utils import set_seed, get_time_diff, try_all_gpus
from data_io.data_parser import read_labels, DataParser
from sklearn import metrics
from torch import nn
import numpy as np
from torch.nn import functional as F
import time
from tensorboardX import SummaryWriter
from d2l import torch as d2l


class ClassificationModelHandler(object):
    def __init__(self, model_name=None, data_path=None, epochs=10, batch_size=256, learning_rate=1e-4,
                 learning_rate_period=2,
                 learning_rate_decay=0.9, weight_decay=1e-4, seed=1):
        """
        init basic params
        :param model_name:
        :param data_path:
        :param epochs: num of iter in all train data
        :param batch_size:
        :param learning_rate:
        :param weight_decay:
        :param learning_rate_period:
        :param learning_rate_decay:
        :param seed:
        :return:
        """
        self._model_name = model_name
        self._data_path = data_path
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._learning_rate_period = learning_rate_period
        self._learning_rate_decay = learning_rate_decay
        self._weight_decay = weight_decay
        self._seed = seed
        self._class_list = read_labels(self._data_path).values()
        self._class_label = sorted(set(self._class_list))
        self._log_path = os.path.join('result', 'log', self._model_name)
        self._save_path = os.path.join('result', 'saved_dict', self._model_name + '.ckpt')
        self._require_improvement = 3000  # 若超过3000次效果没有提升，则提前结束训练
        self._devices = try_all_gpus()

    def build_model(self):
        net = import_module(name='model_zoo.' + self._model_name)
        return net.Model(self._devices)

    def prepare_data(self, data_path, batch_size):
        """
        将数据集转变成直接运行的迭代器
        :param data_path:
        :param batch_size:
        :return:
        """
        train_compose, valid_test_compose = self.build_transform_compose()

        train_folder = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'train_valid_test', 'train'),
                                                        transform=train_compose)
        train_valid_folder = torchvision.datasets.ImageFolder(
            root=os.path.join(data_path, 'train_valid_test', 'train_valid'), transform=train_compose)
        valid_folder = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'train_valid_test', 'valid'),
                                                        transform=valid_test_compose)
        test_folder = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'train_valid_test', 'test'),
                                                       transform=valid_test_compose)
        # drop_last 如果最后一个batch的样本数量不够就丢弃
        train_iter = torch.utils.data.DataLoader(train_folder, batch_size=batch_size, shuffle=True, drop_last=True)
        train_valid_iter = torch.utils.data.DataLoader(train_valid_folder, batch_size=batch_size, shuffle=True,
                                                       drop_last=True)
        valid_iter = torch.utils.data.DataLoader(valid_folder, batch_size=batch_size, shuffle=False, drop_last=True)
        test_iter = torch.utils.data.DataLoader(test_folder, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_iter, train_valid_iter, valid_iter, test_iter

    def build_transform_compose(self):
        train_compose = torchvision.transforms.Compose([
            # 随机裁剪图像，所得图像为原始面积的0.08到1之间，高宽比在3/4和4/3之间。然后缩放图像以创建224x224的新图像
            torchvision.transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            torchvision.transforms.RandomHorizontalFlip(),  # 随机图像左右翻转
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # 亮度，对比因子，饱和度，色调
            torchvision.transforms.ToTensor(),  # 随机噪声
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 来自ImageNet的大规模训练
        ])
        valid_test_compose = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=256),
            # 从图像中心裁切224x224大小的图片
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return train_compose, valid_test_compose

    def train(self):
        # 先设置随机种子，保证运行结果保持一致
        set_seed(seed=self._seed)
        start_time = time.time()
        if not os.path.exists(os.path.join(self._data_path, 'train_valid_test', 'test')):
            DataParser.data_parser(self._data_path, valid_ratio=0.1)
        train_iter, train_valid_iter, valid_iter, test_iter = self.prepare_data(data_path=self._data_path,
                                                                                batch_size=self._batch_size)
        model = self.build_model()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self._learning_rate_period,
                                                       gamma=self._learning_rate_decay)


        loss = nn.CrossEntropyLoss(reduction='none')
        total_batch = 0
        best_loss = float('inf')
        last_improve = 0
        flag = True
        writer = SummaryWriter(log_dir=self._log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
        # for epoch in range(self._epochs):
        #     print('Epoch [{}/{}]'.format(epoch + 1, self._epochs))
        #     model.train()
        #     pred_all = []
        #     true_all = []
        #     for i, (feature, label) in enumerate(train_iter, 0):
        #         train_loss, train_acc, y_pred, y_true = self.train_batch(model, feature, label, loss, optimizer,
        #                                                                  self._devices)
        #         pred_all = np.append(pred_all, y_pred)
        #         true_all = np.append(true_all, y_true)
        #         val_acc, val_loss = self.evaluate(model, valid_iter, self._devices)
        #         if total_batch % 100 == 0:
        #             if val_loss < best_loss:
        #                 best_loss = val_loss
        #                 if not os.path.exists(self._save_path):
        #                     os.mknod(self._save_path)
        #                 torch.save(model.state_dict(), self._save_path)
        #                 improve = '*'
        #                 last_improve = total_batch
        #             else:
        #                 improve = ''
        #             time_diff = get_time_diff(start_time=start_time)
        #             msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2}, Val Acc: {4:>6.2%},  Time: {5} {6}'
        #             current_train_acc = metrics.accuracy_score(pred_all, true_all)
        #             print(msg.format(total_batch, train_loss, current_train_acc, val_loss, val_acc, time_diff, improve))
        #             writer.add_scalar('loss/train', train_loss, total_batch)
        #             writer.add_scalar("loss/dev", val_loss, total_batch)
        #             writer.add_scalar("acc/train", current_train_acc, total_batch)
        #             writer.add_scalar("acc/dev", val_acc, total_batch)
        #         total_batch += 1
        #         if total_batch - last_improve > self._require_improvement:
        #             # 验证集loss超过1000batch没下降，结束训练
        #             print("No optimization for a long time, auto-stopping...")
        #             flag = True
        #             break
        #     if flag:
        #         break
        #     lr_scheduler.step()
        # writer.close()
        # 测试集运行
        self.test(model, test_iter, self._devices)

    def train_batch(self, net, X, y, loss, trainer, devices):
        """Train for a minibatch with mutiple GPUs (defined in Chapter 13).

        Defined in :numref:`sec_image_augmentation`"""
        if isinstance(X, list):
            X = [x.to(devices[0]) for x in X]
        else:
            X = X.to(devices[0])
        y = y.to(devices[0])
        net.train()
        trainer.zero_grad()
        pred = net(X)
        l = loss(pred, y)
        l.sum().backward()
        trainer.step()
        train_loss_sum = l.sum()
        train_acc = metrics.accuracy_score(y.data.cpu().numpy(), d2l.argmax(pred, axis=1).data.cpu().numpy())
        return train_loss_sum, train_acc, d2l.argmax(pred, axis=1).data.cpu().numpy(), y.data.cpu().numpy()

    def test(self, model: nn.Module, test_iter, devices):
        model.load_state_dict(state_dict=torch.load(self._save_path))
        model.eval()
        start_time = time.time()
        test_acc, test_loss, test_report, test_confusion = self.evaluate(model=model, data_iter=test_iter,
                                                                         devices=devices, test=True)
        msg = 'Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print('Precision, Recall and F1-Score...')
        print(test_report)
        print('Confusion Matrix...')
        print(test_confusion)
        time_diff = get_time_diff(start_time)
        print('Time usage: ', time_diff)

    def evaluate(self, model: nn.Module, data_iter: torch.utils.data.DataLoader, devices: list, test=False):
        model.eval()  # 不更新参数
        total_loss = 0
        predict_all = np.array(object=[], dtype=int)
        label_all = np.array(object=[], dtype=int)
        preds = []
        with torch.no_grad():
            for data, label in data_iter:
                data = data.to(devices[0])
                label = label.to(devices[0])
                output = model(data)
                loss = F.cross_entropy(input=output, target=label)
                total_loss += loss
                label = label.data.cpu().numpy()
                predict = torch.max(output.data, 1)[1].cpu().numpy()
                predict_all = np.append(arr=predict_all, values=predict)
                label_all = np.append(arr=label_all, values=label)
                preds.extend(F.softmax(output, dim=0).cpu().detach().numpy())
        if test:
            ids = sorted(os.listdir(os.path.join(self._data_path, 'train_valid_test', 'test', 'unknown')))
            with open('submission1.csv', 'w') as f:
                f.write('id,' + ','.join(self._class_label) + '\n')
                for i, ouput in zip(ids, preds):
                    f.write(i.split('.')[0] + ',' + ','.join([str(num) for num in output]) + '\n')
        acc = metrics.accuracy_score(y_true=label_all, y_pred=predict_all)
        if test:
            report = metrics.classification_report(y_true=label_all, y_pred=predict_all, target_names=self._class_label,
                                                   digits=4)
            confusion = metrics.confusion_matrix(y_true=label_all, y_pred=predict_all)
            return acc, total_loss / len(data_iter), report, confusion
        return acc, total_loss / len(data_iter)
