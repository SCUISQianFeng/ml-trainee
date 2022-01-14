#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng 
@license: Apache Licence 
@file: law_event_extract_pipeline.py 
@time: 2022/01/10
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm 
"""
import re
import os
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
import csv
from utils import label_case, get_event_elements, get_patterns_from_dict, extract_seg

# 1 文本数据预处理

punctuation = '''，。、：；（）ＸX*xa"“”,<《》'''
with open(file='./data/single_case.txt', mode='r', encoding='utf-8') as f1:
    content = f1.read().strip()
    line1 = re.sub(pattern=u'（.*?）', repl='', string=content)
    line2 = re.sub(pattern=u'[{}]+'.format(punctuation), repl='', string=content)
    f2 = open(file='./data/single_preprocess_data.txt', mode='w', encoding='utf-8')
    f2.write(line2)
    f2.close()
#
# f2 = open(file='./data/single_preprocess_data.txt', mode='r', encoding='utf-8')
# content = f2.read()
# # 2 分词+词性标注+命名实体识别
# # 词性附录表  http://ltp.ai/docs/appendix.html
# ltp_data_path = r'E:\DataSet\pretrained\segment\ltp_data_v3.4.0'
# cws_model_path = os.path.join(ltp_data_path, 'cws.model')
# pos_model_path = os.path.join(ltp_data_path, 'pos.model')
# ner_model_path = os.path.join(ltp_data_path, 'ner.model')
#
# segmentor = Segmentor(model_path=cws_model_path)
# postagger = Postagger(model_path=pos_model_path)
# recognizer = NamedEntityRecognizer(ner_model_path)
#
# seg_list = segmentor.segment(content)
# postags = postagger.postag(seg_list)
# ner_list = recognizer.recognize(seg_list, postags)
#
# f3 = open('./data/seg_pos_ner.txt', mode='w', encoding='utf-8')
# for word, pos, ner in zip(seg_list, postags, ner_list):
#     if word == '\n':
#         f3.write('\n')
#     else:
#         f3.write(word + ' ' + pos + ' ' + ner + '\n')
# f2.close()
# f3.close()
# segmentor.release()
# postagger.release()
# recognizer.release()
# #
# # 3.crf提取特征
# os.system(r'.\crf\crf_test.exe -m .\crf\model .\data\seg_pos_ner.txt >> .\data\crf_result.txt')

# 4.提取标签
headers = [
    '01死亡人数',
    "02重伤人数",
    "04责任认定",
    "05是否酒后驾驶",
    "06是否吸毒后驾驶",
    "07是否无证驾驶",
    "08是否无牌驾驶",
    "09是否不安全驾驶",
    "10是否超载",
    "11是否逃逸",
    "12是否抢救伤者",
    "13是否报警",
    "14是否现场等待",
    "15是否赔偿",
    "16是否认罪",
    "18是否初犯偶犯",
    "判决结果"]
rows = []

labels = label_case('./data/single_preprocess_data.txt', is_label=True)
num_classes = 1

f1 = open('./data/single_preprocess_data.txt', 'r', encoding='utf-8')
cases = f1.readlines()

event_elements = get_event_elements('./data/crf_result.txt')
patterns = get_patterns_from_dict(event_elements)
patterns['01死亡人数'], patterns['02重伤人数'], patterns['03轻伤人数'] = extract_seg(line2)
del patterns["是否初犯偶犯"]
del patterns["03轻伤人数"]
del patterns["17是否如实供述"]
rows.append(patterns)
f1.close()

# 写回数据
with open('./data/pattern.csv', 'w', newline='') as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writerows(rows)

f3 = open('./data/pattern.csv', 'r')
f3_csv = list(csv.reader(f3))

f4 = open('./data/feature_extract_data.csv', 'r')
f4_csv = list(csv.reader(f4))
mark = []

# NM，判断相似是通过对比两个案件16
for i in range(len(f4_csv)):
    if f3_csv[0] == f4_csv[i]:
        mark.append(i)

f3.close()
f4.close()

with open("./data/sim_case.txt", "w", encoding="utf-8") as f5:
    f6 = open("./data/all_preprocessed_data.txt", "r", encoding="utf-8")
    contents = f6.readlines()
    for i in range(len(mark)):
        f5.write(contents[i])
    f6.close()




if __name__ == '__main__':
    # str_path = r'./data/crf_result.txt'
    # merge_element(str_path)
    str1 = '本院认为:被告人李俊违反交通运输管理法规行车肇事致一人死亡三人重伤五人轻伤负事故全部责任其行为已构成交通肇事罪应予惩处公诉机关指控的事实和罪名成立本院予以支持被告人李俊犯罪后自动投案如实供述了自己的罪行依法可从轻或减轻处罚其赔偿了被害方经济损失并取得了谅解可对其酌情从轻处罚依照中华人民共和国刑法第一百三十三条第六十七条第一款第七十二条第一款第七十三条第二三款及最高人民法院关于审理交通肇事刑事案件具体应用法律若干问题的解释第二条第一款第一项的规定判决如下:被告人李俊犯交通肇事罪判处有期徒刑九个月缓刑一年六个月缓刑考验期限从判决确定之日起计算'
    # print(extract_seg(str1))

