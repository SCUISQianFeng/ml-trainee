#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: nlpir_segment.py 
@time: 2022/01/04
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

import pynlpir

if __name__ == '__main__':
    pynlpir.open()
    with open(file='./data/test.txt', mode='r', encoding='utf8') as f2:
        line = f2.read()
        result = pynlpir.segment(line, 0)
        with open(file='./data/nlpir_precessed.txt', mode='w', encoding='utf8') as f3:
            for word in result:
                f3.write(str(word) + " ")

    pynlpir.close()

