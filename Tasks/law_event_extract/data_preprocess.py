#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: data_preprocess.py 
@time: 2022/01/04
@contact: scuisqianfeng@gmail.com
@site:  
@software: PyCharm 
"""
import re
from random import shuffle

# 定义标点符号和特殊字母
punctuation = '''，。、:；（）ＸX×xa"“”,<《》'''


def data_process():
    data_path = './data/original_data.txt'
    process_cases = []
    with open(file=data_path, mode='r', encoding='utf8') as f1:
        for line in f1.readlines():
            try:
                location, content = line.strip().split('\t')
            except ValueError:  # ValueError: not enough values to unpack (expected 2, got 1)
                continue
            line1 = re.sub(pattern=u"（.*?）", repl="", string=content)  # 去除末尾括号中的内容
            line2 = re.sub(pattern=u"[{punctuation}]+".format(punctuation=punctuation), repl="",
                           string=line1)  # 去除标点符号、特殊字符
            line3 = re.sub(
                pattern=u"本院认为|违反道路交通管理法规|驾驶机动车辆|因而|违反道路交通运输管理法规|违反交通运输管理法规|缓刑考.*?计算|刑期.*?止|依照|《.*?》|第.*?条|第.*?款|的|了|其|另|已|且",
                repl="", string=line2)
            # 删除内容过少或过长的文书，删除包含’保险‘的文书，只保留以’被告人‘开头的文书
            if 100 < len(line3) < 400 and line3.startswith("被告人") and "保险" not in line3:
                process_cases.append(location + "\t" + line3)

    shuffle(process_cases)  # 打乱数据

    # 预处理的数据写入文本中
    output_path = './data/processed_data.txt'
    with open(file=output_path, mode='w', encoding='utf8') as f2:
        for idx, case in enumerate(process_cases):
            f2.write(str(idx + 1) + "\t" + case + "\n")


if __name__ == '__main__':
    data_process()
