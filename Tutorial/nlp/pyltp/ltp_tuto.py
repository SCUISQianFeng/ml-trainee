#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: ltp_tuto.py 
@time: 2022/01/05
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
    # 分词
    segmentor = Segmentor(model_path=cws_model_path)  # 初始化分词实例
    words = segmentor.segment('他叫汤姆去拿外衣。')
    print(type(words))
    print('\t'.join(words))

    # 词性标注
    postagger = Postagger(model_path=pos_model_path)
    postags = postagger.postag(words)
    print('\t'.join(postags))

    # 语义依存分析
    parser = Parser(parser_model_path)
    arcs = parser.parse(words, postags)
    print("\t".join("%d:%s" % (head, relation) for (head, relation) in arcs))

    # --------------------- 命名实体识别 ------------------------
    recognizer = NamedEntityRecognizer(ner_model_path)
    netags = recognizer.recognize(words, postags)
    print("\t".join(netags))

    # --------------------- 语义角色标注 ------------------------
    labeller = SementicRoleLabeller(pisrl_model_path)
    roles = labeller.label(words, postags, arcs)
    for index, arguments in roles:
        print(index, " ".join(["%s: (%d,%d)" % (name, start, end) for (name, (start, end)) in arguments]))

    segmentor.release()
    segmentor.release()
    postagger.release()
    parser.release()
    recognizer.release()
    labeller.release()

