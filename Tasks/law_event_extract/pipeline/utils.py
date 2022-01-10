#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: utils.py 
@time: 2022/01/10
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""

import re


def remove_duplicate_elements(elements):
    new_list = []
    for ele in elements:
        if ele not in new_list:
            new_list.append(ele)
    return new_list


def find_element(l, *ss):
    """
    查找在l的元素中中是否包含s
    :param l:列表
    :param ss:一个或多个字符串
    :return:
    """
    for s in ss:
        for element in l:
            if s in element:
                return '1'
    return '0'


def text2num(text):
    num = 0
    text = "".join(text)
    digit = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if text:
        idx_q, idx_b, idx_s = text.find('千'), text.find('百'), text.find('十')
        if idx_q != -1:
            num += digit[text[idx_q - 1: idx_q]] * 1000
        if idx_b != -1:
            num += digit[text[idx_b - 1: idx_b]] * 100
        if idx_s != -1:
            # 十前忽略一的处理, 默认1
            num += digit.get(text[idx_s - 1: idx_s], 1) * 10
        if text[-1] in digit:
            num += digit[text[-1]]
    return num


def per_num(text):
    string = re.findall(r"\d+", text)
    if len(string) == 0:
        r1 = re.compile(u'[一二两三四五六七八九十]{1,}')
        r2 = r1.findall(text)
        if len(r2) == 0:
            num = 1
        else:
            num = text2num(r2)
    else:
        num = string[0]
    return num


def extract_seg(content):
    # 死亡人数、重伤人数、轻伤人数提取
    r1 = re.compile(pattern=u'[123456789一二两三四五六七八九十 ]*人( )*死亡')
    r2 = re.search(pattern=r1, string=content)
    if r2 is None:
        num1 = 0
    else:
        text = r2.group()
        num1 = per_num(text)
    # 重伤人数
    r3 = re.compile(pattern=u'[123456789一二两三四五六七八九十 ]*人( )*重伤')
    r4 = re.search(pattern=r3, string=content)
    if r4 is None:
        num2 = 0
    else:
        text = r4.group()
        num2 = per_num(text)
    # 受伤人数
    r5 = re.compile(pattern=u'[123456789一二两三四五六七八九十 ]*人( )*轻伤')
    r6 = re.search(pattern=r5, string=content)
    if r6 is None:
        num3 = 0
    else:
        text = r6.group()
        num3 = per_num(text)
    return num1, num2, num3


def sentence_result(text):
    text = text.strip(" ")
    text = text.replace(" ", "")
    if text.find("判决如下") != -1:
        result = text.split("判决如下")[-1]
    elif text.find("判处如下") != -1:
        result = text.split("判处如下")[-1]
    else:
        result = text
    r1 = re.compile(pattern=u'(有期徒刑|拘役)[一二三四五六七八九十又年零两]{1,}(个月|年)')
    r2 = re.search(r1, result)
    if r2 is None:
        num = 0
    else:
        text = r2.group()
        r3 = re.compile(u'[一二三四五六七八九十两]{1,}')
        r4 = r3.findall(text)
        if len(r4) > 1:
            num1 = text2num(r4[0])
            num2 = text2num(r4[1])
            num = 12 * num1 + num2
        elif text.find(u"年") != -1:
            num = 12 * text2num(r4)
        else:
            num = text2num(r4)
    return num


# 案件要素提取
def get_event_elements(file_name: str = r'./data/crf_result.txt'):
    words = []
    element_types = []
    with open(file='./data/crf_result.txt', mode='r', encoding='utf-8') as f1:
        rows = []
        lines = f1.readlines()
        for line in lines:
            rows.append(line.strip('\n').split('\t'))

        for index, row in enumerate(rows):
            if 'S' in row[-1]:  # 单个元素直接处理
                words.append(row[0])
                element_types.append(row[-1][-1])
            elif 'B' in row[-1]:
                words.append(row[0])
                element_types.append(row[-1][-1])
                j = index + 1
                while 'I' in rows[j][-1] or 'E' in rows[j][-1]:
                    words[-1] += rows[j][0]
                    j += 1
                    if j == len(rows):
                        break
        # 将事件要素进行分类（将words列表中的元素按照类别分成6类）
        T, K, D, P, N, R = [], [], [], [], [], []  # 事故类型, 罪名, 主次责任, 积极因素（减刑因素）,消极因素（加刑因素）,判决结果

        for i in range(len(element_types)):
            if element_types[i] == 'T':
                T.append(words[i])
            elif element_types[i] == "K":
                K.append(words[i])
            elif element_types[i] == "D":
                D.append(words[i])
            elif element_types[i] == "P":
                P.append(words[i])
            elif element_types[i] == "N":
                N.append(words[i])
            elif element_types[i] == "R":
                R.append(words[i])
    # 为了防止CRF未能抽取出全部的事件要素，因此使用规则化的方法，从原始文本中直接提取出部分事件要素，作为补充
    case = ""  # case是完整的案件内容
    for index in range(len(rows)):
        case += rows[index][0]

    if '无证' in case or '驾驶资格' in case:
        N.append('无证驾驶')
    if '无号牌' in case or '拍照' in case or '无牌' in case:
        N.append('无牌驾驶')
    if '酒' in case:
        N.append('酒后驾驶')
    if '吸毒' in case or '毒品' in case or '毒驾' in case:
        N.append('想吸毒后驾驶')
    if '超载' in case:
        N.append('超载')
    if '逃逸' in case or '逃离' in case:
        N.append('逃逸')
    if ('有前科' in case or '有犯罪前科' in case) and ('无前科' not in case and '无犯罪前科' not in case):
        N.append('有犯罪前科')

    # 整理抽取结果
    event_elements = dict()  # 用字典存储各类事件要素
    event_elements["事故类型"] = remove_duplicate_elements(T)
    event_elements["罪名"] = remove_duplicate_elements(K)
    event_elements["主次责任"] = remove_duplicate_elements(D)
    event_elements["减刑因素"] = remove_duplicate_elements(P)
    event_elements["加刑因素"] = remove_duplicate_elements(N)
    event_elements["判决结果"] = remove_duplicate_elements(R)

    # 打印出完整的时间要素
    for key, value in event_elements.items():
        print(key, " => ", value)
    return event_elements


def get_patterns_from_dict(event_elements):
    # 从事件要素中的“加刑因素”提取出三个特征：01死亡人数，02重伤人数，03轻伤人数
    patterns = dict()
    patterns['01死亡人数'], patterns['02重伤人数'], patterns['03轻伤人数'] = extract_seg("".join(event_elements["加刑因素"]))
    # 从事件要素中的"主次责任"提取出特征：04责任认定
    patterns['04责任认定'] = find_element(event_elements['主次责任'], '全部责任')
    patterns["07是否无证驾驶"] = find_element(event_elements["加刑因素"], "驾驶证", "证")
    patterns["08是否无牌驾驶"] = find_element(event_elements["加刑因素"], "牌照", "牌")
    patterns["09是否不安全驾驶"] = find_element(event_elements["加刑因素"], "安全")
    patterns["10是否超载"] = find_element(event_elements["加刑因素"], "超载")
    patterns["11是否逃逸"] = find_element(event_elements["加刑因素"], "逃逸", "逃离")
    patterns["是否初犯偶犯"] = 1 - int(find_element(event_elements["加刑因素"], "前科"))

    # 从事件要素中的"减刑因素"提取出7个特征
    patterns["12是否抢救伤者"] = find_element(event_elements["减刑因素"], "抢救", "施救")
    patterns["13是否报警"] = find_element(event_elements["减刑因素"], "报警", "自首", "投案")
    patterns["14是否现场等待"] = find_element(event_elements["减刑因素"], "现场", "等候")
    patterns["15是否赔偿"] = find_element(event_elements["减刑因素"], "赔偿")
    patterns["16是否认罪"] = find_element(event_elements["减刑因素"], "认罪")
    patterns["17是否如实供述"] = find_element(event_elements["减刑因素"], "如实")
    if patterns["是否初犯偶犯"] == 0:
        patterns["18是否初犯偶犯"] = "0"
    else:
        patterns["18是否初犯偶犯"] = "1"
    return patterns


def label_case(file, is_label=False):
    f = open(file, 'r', encoding='utf-8')
    cases = f.readlines()
    labels = []
    for case in cases:
        labels.append(sentence_result(case))
    f.close()

    # 分类
    if is_label:
        for i in range(len(labels)):
            if 0 <= labels[i] <= 5:  # 刑期半年内
                labels[i] = 0
            elif 6 <= labels[i] <= 18:
                labels[i] = 1
            elif 19 <= labels[i] <= 24:
                labels[i] = 2
            elif 25 <= labels[i] <= 36:
                labels[i] = 3
            else:
                labels[i] = 4
    return labels
