#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: extract_result.py 
@time: 2022/01/09
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
import os
import json


def remove_duplicate_elements(l):
    """直接list转set不香"""
    new_list = []
    for i in l:
        if i not in new_list:
            new_list.append(i)
    return new_list


def merge_element(file_name):
    words = []
    element_type = []
    with open(file_name, 'r', encoding='utf-8') as f1:
        contents = f1.readlines()
        new_contents = []
        for content in contents:
            new_contents.append(content.strip("\n").split(' '))
        for index, content in enumerate(new_contents):
            if 'S' in content[-1]:
                # 处理由一个单词组成的事件要素
                words.append(contents[0])
                element_type.append(content[-1])
            elif 'B' in content[-1]:
                # 处理由多个单词组成的事件要素
                words.append(content[0])
                element_type.append(content[-1])
                j = index + 1
                while 'I' in new_contents[j][-1] or 'E' in new_contents[j][-1]:
                    words[-1] = words[-1] + new_contents[j][0]  # 多个单词拼在一起
                    j += 1
                    if j == len(new_contents):
                        break
        T, K, D, P, N, R = [], [], [], [], [], []
        for i in range(len(element_type)):
            if element_type[i][-1] == 'T':
                T.append(words[i])
            elif element_type[i][-1] == "K":
                K.append(words[i])
            elif element_type[i][-1] == "D":
                D.append(words[i])
            elif element_type[i][-1] == "P":
                P.append(words[i])
            elif element_type[i][-1] == "N":
                N.append(words[i])
            elif element_type[i][-1] == "R":
                R.append(words[i])
        # 整理抽取结果
        result = dict()
        result["事故类型"] = remove_duplicate_elements(T)
        result["罪名"] = remove_duplicate_elements(K)
        result["主次责任"] = remove_duplicate_elements(D)
        result["减刑因素"] = remove_duplicate_elements(P)
        result["加刑因素"] = remove_duplicate_elements(N)
        result["判决结果"] = remove_duplicate_elements(R)
    return result




if __name__ == '__main__':
    # 切割数据
    # with open(file='./data/output.txt', mode='r', encoding='utf-8') as f1:
    #     data = f1.readlines()
    #     num = 1
    #     file_name = 'meta_data' + str(num) + '.txt'
    #     for i in range(len(data)):
    #         f2 = open(file_name, 'w', encoding='utf-8')
    #         if data[i] == '\n':
    #             num += 1
    #             file_name = os.path.join('meta_data', str(num) + '.txt')
    #         else:
    #             f2.writelines(data[i])
    # 事件要素处理
    print(merge_element('./meta_data/1.txt'))


    # 减刑/加刑
    dd = dict()
    txt_num = 694+1
    for i in range(1, txt_num):
        file_name = os.path.join('.data', str(i) + '.txt')
        dd[i] = merge_element(file_name)

    with open(file='element_statistics.json', mode='w', encoding='utf-8') as f:
        json.dump(dd, f, ensure_ascii=False)

    # 减刑因素规范化
    positive = []
    for i in range(1, txt_num):
        positive += dd[i]["减刑因素"]
    for i in range(len(positive)):
        if "自首" in positive[i] or "投案" in positive[i]:
            positive[i] = "自首"
        elif "谅解" in positive[i] or "取得" in positive[i]:
            positive[i] = "取得谅解"
        elif "赔偿" in positive[i]:
            positive[i] = "赔偿被害人"
        elif "如实供述" in positive[i] or "坦白" in positive[i]:
            positive[i] = "如实供述"
        elif "认罪" in positive[i]:
            positive[i] = "自愿认罪"

    # 加刑因素规范化
    negative = []
    for i in range(1, txt_num):
        negative += dd[i]["加刑因素"]
    for i in range(len(negative)):
        if "逃" in negative[i]:
            negative[i] = "逃逸"
        elif "无证" in negative[i] or "驾驶资格" in negative[i] or "驾驶证" in negative[i] or "无牌" in negative[i]:
            negative[i] = "无牌或无证驾驶"
        elif "酒" in negative[i]:
            negative[i] = "酒后驾驶"
        elif "超速" in negative[i]:
            negative[i] = "超速"
