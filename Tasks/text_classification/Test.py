#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuislishuai 
@license: Apache Licence 
@file: Test.py 
@time: 2021/12/16
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
import argparse
import os
import pickle as pkl
import time
from datetime import timedelta
import numpy as np

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from tensorboardX import SummaryWriter

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


########################################################
# 构建数据集
########################################################
def build_vocab(file_path: str, tokenizer, max_size: int, min_seq: int) -> dict:
    """
    构建词典
    :param file_path: 构造词典的原始文件路径
    :param tokenizer: 分词器
    :param max_size: 词表的最大长度
    :param min_seq: 词表中字的最小出现次数
    :return: 字典：dict
    """
    vocab_dict = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content, label = line.split('\t')
            for word in tokenizer(content):
                vocab_dict[word] = vocab_dict.get(word, 0) + 1  # 默认值是0
        vocab_list = sorted([word_tuple for word_tuple in vocab_dict.items() if word_tuple[1] > min_seq],
                            key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dict = {word_tuple[0]: idx for idx, word_tuple in enumerate(vocab_list)}
        vocab_dict.update({UNK: len(vocab_dict), PAD: len(vocab_dict) + 1})
        return vocab_dict


def build_dataset(config: object, use_word: bool) -> (dict, list, list, list):
    """
    构建词典， 训练集，验证集，测试集
    :param config:
    :param use_word:
    :return: （词典， 训练集，验证集，测试集）
    """
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]
    if os.path.exists(config.vocab_path):
        vocab_dict = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab_dict = build_vocab(config.train_path, tokenizer=tokenizer, max_size=config.MAX_VOCAB_SIZE, min_seq=1)
        pkl.dump(vocab_dict, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab_dict)}")

    def load_dataset(path, pad_size):
        """
        加载数据
        :param path: 文件路径
        :param pad_size: 序列最大长度
        :return: 文本数据word2idx，形如[([x1,x2,..,xn], label, len)]
        """
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                content, label = line.strip().split('\t')
                word_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if (seq_len < pad_size):
                    token.extend([PAD] * (pad_size - seq_len))  # 不够的补长度, 标记出数据的真实长度
                else:
                    token = token[:pad_size]  # 截断
                    seq_len = pad_size
                for word in token:
                    word_line.append(vocab_dict.get(word, vocab_dict.get(UNK)))
                contents.append((word_line, int(label), seq_len))  # ([word2idx, label, seq_len)]
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab_dict, train, dev, test


class DatasetIterator(object):
    def __init__(self, batches, batch_size, device):
        self.batches = batches
        self.batch_size = batch_size
        self.device = device
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0

    def __next__(self):
        if self.residue and self.index == self.n_batches:  # 最后一轮迭代，不够一个batch_size的数据
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            batches = self._to_tensor(batches)
            self.index += 1
            return batches
        elif self.index >= self.n_batches:  # 正常迭代完，重置迭代标识
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batches = self._to_tensor(batches)
            self.index += 1
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        """
        迭代器的循环轮数
        :return:
        """
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)  # ([word2idx, label, seq_len)][0] -> word2idx
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)  # ([word2idx, label, seq_len)][1] -> label
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)  # ([word2idx, label, seq_len)][2] -> 真实序列长度
        return (x, seq_len), y


def build_iterator(dataset, config):
    return DatasetIterator(dataset, config.batch_size, config.device)


def get_time_diff(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))


########################################################
# 训练过程
########################################################
def train(config, model: nn.Module, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        scheduler.step()
        for idx, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(input=outputs, target=labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predict = torch.max(outputs.data, dim=1)[1].cpu()
                train_acc = metrics.accuracy_score(y_true=trains, y_pred=predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_best_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_diff = get_time_diff(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_diff, improve))
                writer.add_scalar(tag='loss/train', scalar_value=loss.item(), global_step=total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch = 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model: nn.Module, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_diff(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()  # 不更新参数
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(input=outputs, target=labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, dim=1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(y_true=labels_all, y_pred=predict_all)
    if test:
        report = metrics.classification_report(y_true=labels_all, y_pred=predict_all, target_names=config.class_list,
                                               digits=4)
        confusion = metrics.confusion_matrix(y_true=labels_all, y_pred=predict_all)
        return acc, loss_total / len(data_iter), report, config
    return acc, loss_total / len(data_iter)

    pass


class Config:
    def __init__(self):
        self.vocab_path = './THUCNews/data/vocab.pkl'
        self.train_path = './THUCNews/data/train.txt'
        self.dev_path = './THUCNews/data/dev.txt'
        self.test_path = './THUCNews/data/test.txt'
        self.pad_size = 32
        self.MAX_VOCAB_SIZE = MAX_VOCAB_SIZE


if __name__ == '__main__':
    # 测试词表构方法
    config = Config()
    vocab_dict, train, dev, test = build_dataset(config, use_word=False)
    print('pass')
    # pkl.dump(vocab_didct, open('./THUCNews/data/vocab1.pkl', 'wb'))
