#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: ltp_segment.py 
@time: 2022/01/04
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

from pyltp import Postagger
from pyltp import Segmentor
from pyltp import Parser
from pyltp import NamedEntityRecognizer
from pyltp import SementicRoleLabeller
import os

LTP_DATA_DIR = r'E:\DataSet\ltp_data_v3.4.0'
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
parser_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
pisrl_model_path = os.path.join(LTP_DATA_DIR, 'pisrl_win.model')

if __name__ == '__main__':
    with open(file='./data/preprocessed_data1.txt', mode='r', encoding='utf-8') as f1:
        content = f1.read()
        # 分词
        segmentor = Segmentor(model_path=cws_model_path, lexicon_path='./data/dict')  # 初始化分词实例
        seg_list = segmentor.segment(content)
        seg_list = list(seg_list)
        # LTP不能很好地处理回车，因此需要去除回车给分词带来的干扰。
        # LTP也不能很好地处理数字，可能把一串数字分成好几个单词，因此需要连接可能拆开的数字
        i = 0
        while i < len(seg_list):
            # 如果单词里包含回车，则需要分三种情况处理
            if '\n' in seg_list[i] and len(seg_list[i]) > 1:
                idx = seg_list[i].find('\n')
                # 回车在单词的开头，如\n被告人
                if idx == 0:
                    remains = seg_list[i][1:]
                    seg_list[i] = '\n'
                    seg_list.insert(i + 1, remains)
                # 回车在单词末尾，如被告人\n
                elif idx == len(seg_list[i]) - 1:
                    remains = seg_list[i][:-1]
                    seg_list[i] = remains
                    seg_list.insert(i + 1, '\n')
                else:
                    remains1 = seg_list[i].split('/n')[0]
                    remains2 = seg_list[i].split('/n')[-1]
                    seg_list[i] = remains1
                    seg_list.insert(i + 1, '\n')
                    seg_list.insert(i + 2, remains2)
            # 将拆开的数字连接起来
            if seg_list[i].isdigit() and seg_list[i + 1].isdigit():
                seg_list[i] = seg_list[i] + seg_list[i + 1]
                del seg_list[i + 1]
            i += 1
        # 词性标注
        postagger = Postagger(model_path=pos_model_path)
        postags = postagger.postag(seg_list)

        # 命名实体识别
        recognizer = NamedEntityRecognizer(ner_model_path)
        nertags = recognizer.recognize(seg_list, postags)

        f2 = open(file='./data/seg_pos_ner_result1.txt', mode='w', encoding='utf-8')
        for word, postag, nertag in zip(seg_list, postags, nertags):
            if word == '\n':
                f2.write('\n')
            else:
                f2.write(word + ' ' + postag + ' ' + nertag)
        f2.close()

        # 释放模型
        segmentor.release()
        postagger.release()
        recognizer.release()

