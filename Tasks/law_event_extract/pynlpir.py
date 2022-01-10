#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: pynlpir.py 
@time: 2022/01/05
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

import pynlpir

if __name__ == '__main__':
    pynlpir.open()
    with open(file='./data/test.txt', mode='r', encoding='utf8') as f1:
        text = f1.read()
        result = pynlpir.segment(s=text, pos_tagging=0)
        with open(file='./data/nlpir_precessed.txt', mode='w', encoding='utf8') as f2:
            for word in result:
                f2.write(str(word) + " ")
