#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: data_parser.py
@time: 2022/01/02
@contact: scuisqianfeng@gmail.com
@site:  
@software: PyCharm 
"""
import os
import sys
import shutil
import collections
import math

sys.path.append('..')


def read_labels(data_path):
    """
    读取标签数据
    :param data_path: 数据路径
    :return: {name, label}
    """
    label_path = os.path.join(data_path, 'labels.csv')
    with open(label_path, 'r') as f:
        lines = f.readlines()[1:]  # 去除第一行的标题行
    tokens = [line.strip().split(',') for line in lines]
    return dict(((name, label) for name, label in tokens))


def copy_file(file_name, target_dir):
    """
    将文件从file_name复制到target_dir
    :param file_name:
    :param target_dir:
    :return:
    """
    # target_dir不存在则创建
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(file_name, target_dir)


def split_train_data(data_path: str, labels: dict, valid_ratio=0.1):
    """
    将全部train_data按照valid_ratio分到train_valid, train, valid三个文件夹下
    :param data_path:
    :param labels:
    :param valid_ratio:
    :return:
    """
    # 读取全部数据的label，按照最小类别的数量确定valid文件夹下每个类别的数量
    # most_common()：[(name, num), ..., ],  按照num从大到小排序 [-1][1]就是num最小类别的数量
    n_min_label = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数(向下取整）
    n_label_valid_num = max(1, math.floor(n_min_label * valid_ratio))
    label_count = {}
    for file in os.listdir(os.path.join(data_path, 'train')):
        file_idx = file.split('.')[0]  # 1.jpg => [1, jpg]
        file_label = labels.get(file_idx)
        file_path = os.path.join(data_path, 'train', file)

        # 所有的train_data全部放入train_valid_test\train_valid路径下
        copy_file(file_path, os.path.join(data_path, 'train_valid_test', 'train_valid', file_label))
        if file_label not in label_count or label_count.get(file_label) < n_label_valid_num:
            # 每个类别的数量不够就继续复制
            label_count[file_label] = label_count.get(file_label, 0) + 1
            copy_file(file_path, os.path.join(data_path, 'train_valid_test', 'valid', file_label))
        else:
            copy_file(file_path, os.path.join(data_path, 'train_valid_test', 'train', file_label))
    return n_label_valid_num


def split_test(data_path):
    for test_file in os.listdir(os.path.join(data_path, 'test')):
        copy_file(file_name=os.path.join(data_path, 'test', test_file),
                  target_dir=os.path.join(data_path, 'train_valid_test', 'test', 'unknown'))


class DataParser(object):
    @staticmethod
    def data_parser(data_path, valid_ratio):
        labels = read_labels(data_path=data_path)
        # split_train_data(data_path, labels, valid_ratio)
        split_test(data_path)



