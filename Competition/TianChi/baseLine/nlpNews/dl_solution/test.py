# -*- coding:utf-8 -*-

# -*- coding:utf-8 -*-

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd
# from sklearn.utils import shuffle
import sklearn as sk
import logging
from collections import Counter
# from transformers import BasicTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F

# basic_tokenizer = BasicTokenizer()

# 全局变量设置
# build word encoder
word2vec_path = r'E:\DataSet\Tianchi\nlpNews\word2vec\word2vec.txt'
dropout = 0.15


# 公共静态参数
def reformat(num, n):
    """
    小数格式化 类似于 0.2f
    :param num: 待格式化
    :param n: 保留小数位数
    :return: float
    """
    return float(format(num, '0.' + str(n) + 'f'))


def get_score(y_true, y_pred):
    """
    计算算法准确度
    :param y_true:
    :param y_pred:
    :return:
    """
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro') * 100
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro') * 100
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro') * 100
    return str((reformat(f1, 2), reformat(precision, 2), reformat(recall, 2))), reformat(recall, 2)


def batch_slice(data, batch_size):
    """
    批处理数据切片
    :param data: 源数据
    :param batch_size: 每批的数据大小
    :return: 迭代数据
    """
    print("batch_slice")
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        # 最后一个批次的数据可能小于batch_size
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield docs


def data_iter(data, batch_size, shuffle=True, noise=1.0):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """
    batched_data = []
    if shuffle:
        data = sk.utils.shuffle(data)
        # example[1]: label
        lengths = [example[1] for example in data]
        # 这里的data 是经过编号只有才能使用的
        noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
        sorted_indices = np.argsort(noisy_lengths).tolist()
        sorted_data = [data[i] for i in sorted_indices]
    else:
        sorted_data = data

    batched_data.extend(list(batch_slice(sorted_data, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch


def all_data2fold(data_file, fold_num, num=10000):
    fold_data = []
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    texts = f['text'].tolist()[:num]
    labels = f['label'].tolist()[:num]
    total = len(labels)
    index = list(range(total))
    np.random.shuffle(index)

    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)  # label2id[label]: [1, 4, 8..]
    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # 每个类别分层采样
        batch_size = int(len(data) / fold_num)
        other = len(data) - batch_size * fold_num
        for i in range(fold_num):
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # 在上一批次的基础上采样
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            # 每种类别对应的数据
            all_index[i].extend(batch_data)

    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_labels)

        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)

        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)
    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))

    return fold_data


class Vocab():
    def __init__(self, train_data):
        self.min_count = 5
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']
        self._id2label = []
        self.target_name = []
        self.build_vocab(train_data)

        reverse = lambda x: dict(zip(x, range(len(x))))  # 形如['a', 'b'...] => {'a': 0, 'b': 1,...}
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)

        logging.info("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))

    def build_vocab(self, data):
        self.word_counter = Counter()
        for text in data['text']:
            words = text.split(" ")
            for word in words:
                # 每个词出现的次数
                self.word_counter[word] += 1
        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}
        self.label_counter = Counter(data['label'])

        for label in range(len(self.label_counter)):
            count = self.label_counter[label]
            self._id2label.append(label)
            self.target_name.append(label2name[label])

    def load_pretrained_embs(self, embfile):
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            word_count, embedding_dim = int(items[0]), int(items[1])

        index = len(self._id2extword)  # 单个字长度？
        embeddings = np.zeros((word_count + index, embedding_dim))
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index += 1

        embeddings[self.unk] = embeddings[self.unk] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._exrword2id = reverse(self._id2extword)

        assert len(set(self._id2extword)) == len(self._id2extword)

        return embeddings

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.05)

        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias.data.copy_(torch.from_numpy(b))

        self.query = nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)

    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x len

        # linear
        key = torch.matmul(batch_hidden, self.weight) + self.bias  # b x len x hidden

        # compute attention
        outputs = torch.matmul(key, self.query)  # b x len

        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))

        attn_scores = F.softmax(masked_outputs, dim=1)  # b x len

        # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        # sum weighted sources
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden

        return batch_outputs, attn_scores


if __name__ == "__main__":
    import torch

    a = torch.tensor([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7]], [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]])
    print(a)
    print(a.size())
    print("#############################################3")
    mask = torch.ByteTensor([[[1], [1], [0]], [[0], [1], [1]]])
    print(mask.size())
    b = a.masked_fill(mask, value=torch.tensor(-1e9))
    print(b)
    print(b.size())
    # data_path = r'E:\DataSet\Tianchi\nlpNews\train_set\train_set.csv'
    # # data = pd.read_csv(data_path, sep='\t')
    # data = pd.read_csv(data_path, sep='\t', nrows=100)
    # # build train, dev, test data
    # fold_num = 10
    # data_file = r'E:\DataSet\Tianchi\nlpNews\train_set\train_set.csv'
    # fold_data = all_data2fold(data_file=data_file, fold_num=10)
    #
    # fold_id = 9
    # # dev
    # dev_data = fold_data[fold_id]
    #
    # # train
    # train_texts = []
    # train_labels = []
    # for i in range(0, fold_id):
    #     data = fold_data[i]
    #     train_texts.extend(data['text'])
    #     train_labels.extend(data['label'])
    #
    # train_data = {'label': train_labels, 'text': train_texts}
    #
    # # test
    # test_data_file = r'E:\DataSet\Tianchi\nlpNews\test_set\test_set.csv'
    # f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
    # texts = f['text'].tolist()
    # test_data = {'label': [0] * len(texts), 'text': texts}
    # vocab = Vocab(train_data)
