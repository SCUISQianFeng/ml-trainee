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
from utils import label_case, get_event_elements, get_patterns_from_dict,


# 1 文本数据预处理

# punctuation = '''，。、：；（）ＸX*xa"“”,<《》'''
# with open(file='./data/single_case.txt', mode='r', encoding='utf-8') as f1:
#     content = f1.read().strip()
#     line1 = re.sub(pattern=u'（.*?）', repl='', string=content)
#     line2 = re.sub(pattern=u'[{}]+'.format(punctuation), repl='', string=content)
#     f2 = open(file='./data/preprocess_data.txt', mode='w', encoding='utf-8')
#     f2.write(line2)
#     f2.close()
#
# f2 = open(file='./data/preprocess_data.txt', mode='r', encoding='utf-8')
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



if __name__ == '__main__':
    # str_path = r'./data/crf_result.txt'
    # merge_element(str_path)
    str1 = '本院认为:被告人李俊违反交通运输管理法规行车肇事致一人死亡三人重伤五人轻伤负事故全部责任其行为已构成交通肇事罪应予惩处公诉机关指控的事实和罪名成立本院予以支持被告人李俊犯罪后自动投案如实供述了自己的罪行依法可从轻或减轻处罚其赔偿了被害方经济损失并取得了谅解可对其酌情从轻处罚依照中华人民共和国刑法第一百三十三条第六十七条第一款第七十二条第一款第七十三条第二三款及最高人民法院关于审理交通肇事刑事案件具体应用法律若干问题的解释第二条第一款第一项的规定判决如下:被告人李俊犯交通肇事罪判处有期徒刑九个月缓刑一年六个月缓刑考验期限从判决确定之日起计算'
    # print(extract_seg(str1))
