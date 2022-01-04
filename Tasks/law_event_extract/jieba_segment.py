#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: jieba_segment.py
@time: 2022/01/04
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
import jieba

if __name__ == '__main__':
    with open(file='./data/test.txt', mode='r', encoding='utf8') as f2:
        line = f2.read()
        result = jieba.cut(line)
        with open(file='./data/jieba_precessed.txt', mode='w', encoding='utf8') as f3:
            for word in result:
                f3.write(str(word) + " ")