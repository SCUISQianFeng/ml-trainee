#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: law_similarity.py
@time: 2022/01/04
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
import numpy as np
from gensim.models import word2vec
from scipy import linalg

# hyper parameter
embed_size = 100  # word2vec返回的结果都是100维的向量
sentences = word2vec.LineSentence('./data2/law_text.txt')
model = word2vec.Word2Vec(sentences=sentences, hs=1, min_count=1, window=5, vector_size=100)


def sentence_vector(sentence):
    """将所有单词的词向量相加求平均值，得到的向量即为句子的向量"""
    words = sentence.split(' ')
    result = np.zeros(shape=100)
    for word in words:
        embedding = model.wv[word]  # word对应的向量
        result += embedding
    result /= len(sentence)
    return result


def vector_similarity(sentence1, sentence2):
    """计算两个句子之间的相似度:将两个向量的夹角余弦值作为其相似度"""
    vector1 = sentence_vector(sentence=sentence1)
    vector2 = sentence_vector(sentence=sentence2)
    return np.dot(vector1, vector2) / (linalg.norm(vector1) * linalg.norm(vector2))


if __name__ == '__main__':
    with open(file='./data2/law_text.txt', mode='r', encoding='utf8') as f:
        contents = f.readlines()
        len_con = len(contents)
        matrix_con = np.zeros(shape=(len_con, len_con))
        for i in range(len_con):
            for j in range(len_con):
                matrix_con[i][j] = vector_similarity(contents[i].strip(), contents[j].strip())

        f1 = open(file='./data2/result.txt', mode='w', encoding='utf8')
        for j in range(len_con):
            # 获取最为相似的案件
            # 注意：每个案件与自己的相似度为1，因此获取的是相似度第二大的案件
            index = np.argsort(matrix_con[j])[-2]  # argsort：所有值从小到大排序后在原始数据中的位置

            f1.writelines("案件" + str(j + 1) + ":" + '\t')
            f1.writelines(contents[j])
            f1.writelines("案件" + str(index + 1) + ":" + '\t')
            f1.writelines(contents[index])
            f1.writelines("相似度" + ":" + matrix_con[j][index])
        f1.close()
