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
import pickle as pkl
import os

from tqdm import tqdm

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


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
                contents.append((word_line, int(label), seq_len))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab_dict, train, dev, test


class DatasetIterator(object):
    def __init__(self, batches, batch_size, device):
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
