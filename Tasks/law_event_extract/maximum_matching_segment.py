#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: maximum_matching_segment.py
@time: 2022/01/04
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""


class FMM(object):
    """
        正常最大匹配
        从左往右取词，取词最大长度为词典中长词的长度，每次右边减一个字，直到词典中存在或剩下1个单字
    """

    def __init__(self, dic_path):
        """最大匹配都是基于词典的，每次匹配都需要基于字典中最大长度来运行"""
        self.dictionary = set()  # 构建字典集合
        self.maximum = 0  # 最长单词的长度
        with open(file=dic_path, mode='r', encoding='utf8') as f1:
            for line in f1:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                if len(line) > self.maximum:
                    self.maximum = len(line)

    def cut(self, text):
        result = []
        index = 0
        while index < len(text):
            word = None
            for size in range(self.maximum, 0, -1):
                if len(text) - index < size:
                    # 最后剩下字符长度不足一个size，不做处理
                    continue
                piece = text[index:(index + size)]
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index += size
                    break
            if word is None:
                result.append(text[index])
                index += 1
        return result


class BMM(object):
    """反向最大匹配（BMM）"""

    def __init__(self, dic_path):
        self.dictionary = set()
        self.maximum = 0
        with open(file=dic_path, mode='r', encoding='utf8') as f1:
            for line in f1.readlines():
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                if len(line) > self.maximum:
                    self.maximum = len(line)

    def cut(self, text):
        result = []
        index = len(text)
        while index > 0:
            word = None
            for size in range(self.maximum, 0, -1):
                if index < size:
                    continue
                piece = text[(index - size):index]
                if piece in self.dictionary:
                    word = piece
                    result.append(piece)
                    index -= size
                    break
            if word is None:
                #  走到最后一个字都没有匹配的，就将最后一个字作为分词结果
                result.append(text[index - 1])  # 这里的index需要-1
                index -= 1
        return result[::-1]


class BiMM(object):
    def __init__(self, dic_path):
        self.dictionary = set()
        self.fmm = FMM(dic_path=dic_path)
        self.bmm = BMM(dic_path=dic_path)
        self.maximum = 0
        with open(file=dic_path, mode='r', encoding='utf8') as f1:
            for line in f1.readlines():
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                if len(line) > self.maximum:
                    self.maximum = len(line)

    def cut(self, text):
        """
            首先看两种方法结果的分词数，分词数越少越好；
            分词数相同的情况下，看单个词的数量，越少越好；
        """
        fmm_result = self.fmm.cut(text)
        bmm_result = self.bmm.cut(text)
        if len(fmm_result) < len(bmm_result):
            return fmm_result
        elif len(fmm_result) > len(bmm_result):
            return bmm_result
        else:
            fmm_single = len([word for word in fmm_result if len(word) == 1])
            bmm_single = len([word for word in bmm_result if len(word) == 1])
            # 分词数量相同，单字数量也相同的情况下， 返回哪个都行
            return fmm_single if fmm_single < bmm_single else bmm_single


if __name__ == '__main__':
    tokenize = BiMM(dic_path='./data/dict.txt')
    with open(file='./data/test.txt', mode='r', encoding='utf8') as f2:
        line = f2.read()
        result = tokenize.cut(line)
        with open(file='./data/mm_precessed.txt', mode='w', encoding='utf8') as f3:
            for word in result:
                f3.write(str(word) + " ")

