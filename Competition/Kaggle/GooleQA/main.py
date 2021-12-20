#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: scuislishuai
@license: Apache Licence 
@file: main.py 
@time: 2021/11/08
@contact: scuislishuai@gmail.com
@site:  
@software: PyCharm
"""

import numpy as np

import pandas as pd
import os
import sys
import transformers
import tokenizers
import tensorflow as tf
from tqdm.autonotebook import tqdm

# init parameters
pd.set_option("display.max_columns", None)


class Config:
    MAX_LENGTH = 512
    BERT_PATH = r'E:\DataSet\pretrained\bert'


# const parameters

DATA_PATH = r'E:\DataSet\DataSet\kaggle\kaggle_competition\google-quest-challenge'
TRAIN_DATA_PATH = os.path.join(DATA_PATH + os.path.sep + "train.csv")
TEST_DATA_PATH = os.path.join(DATA_PATH + os.path.sep + "test.csv")

train_df = pd.read_csv(TRAIN_DATA_PATH)

tokenizer = transformers.BertTokenizer(vocab_file=Config.BERT_PATH + os.path.sep + "bert-base-uncased-vocab.txt")


def convert_to_transformer_data(question_title, question_body, question_answer, tokenizer: tokenizer, max_length):
    """
    String -> bert
    :param question_title: 问题标题
    :param question_body: 题干
    :param question_answer: 问题的答案
    :param tokenizer: 分词器
    :param max_length: 输入的最大长度
    :return: ids, attention_ids, ids_type
    """
    title_body = question_title + " " + question_body
    body_answer = question_body + " " + question_answer

    q_input = tokenizer.encode_plus(title_body, text_pair=None, add_special_tokens=True, max_length=max_length)
    a_input = tokenizer.encode_plus(body_answer, text_pair=None, add_special_tokens=True, max_length=max_length)

    q_input_ids = q_input["input_ids"]
    q_tokens_mask = q_input["special_tokens_mask"]
    q_type_ids = q_input["token_type_ids"]
    q_padding = [0] * (max_length - len(q_input_ids))
    q_input_ids = q_input_ids + q_padding
    q_tokens_mask = q_tokens_mask + q_padding
    q_type_ids = q_type_ids + q_padding

    a_input_ids = a_input["input_ids"]
    a_tokens_mask = a_input["special_tokens_mask"]
    a_type_ids = a_input["token_type_ids"]
    a_padding = [0] * (max_length - len(a_input_ids))
    a_input_ids = a_input_ids + a_padding
    a_tokens_mask = a_tokens_mask + a_padding
    a_type_ids = a_type_ids + a_padding

    return q_input_ids, q_tokens_mask, q_type_ids, a_input_ids, a_tokens_mask, a_type_ids


def compute_input_array(df: pd.DataFrame, columns, tokenizer, max_length=Config.MAX_LENGTH):
    q_input_ids_list, q_mask_list, q_token_type_ids_list = [], [], []
    a_input_ids_list, a_mask_list, a_token_type_ids_list = [], [], []

    for _, row in tqdm(df[columns].iterrows()):
        title, body, answer = row.question_title, row.question_body, row.answer
        q_input_ids, q_mask, q_type_ids, a_input_ids, a_mask, a_type_ids = convert_to_transformer_data(title, body,
                                                                                                       answer,
                                                                                                       tokenizer,
                                                                                                       max_length)

        q_input_ids_list.append(q_input_ids)
        q_mask_list.append(q_mask)
        q_token_type_ids_list.append(q_type_ids)

        a_input_ids_list.append(a_input_ids)
        a_mask_list.append(a_mask)
        a_token_type_ids_list.append(a_type_ids)

    return [np.array(q_input_ids_list, dtype=np.int32),
            np.array(q_mask_list, dtype=np.int32),
            np.array(q_token_type_ids_list, dtype=np.int32),
            np.array(a_input_ids_list, dtype=np.int32),
            np.array(a_mask_list, dtype=np.int32),
            np.array(a_token_type_ids_list, dtype=np.int32)
            ]


def model():
    q_ids = tf.keras.Input(shape=(Config.MAX_LENGTH), dtype=tf.int32)
    a_ids = tf.keras.Input(shape=(Config.MAX_LENGTH), dtype=tf.int32)

    q_mask = tf.keras.Input(shape=(Config.MAX_LENGTH), dtype=tf.int32)
    a_mask = tf.keras.Input(shape=(Config.MAX_LENGTH), dtype=tf.int32)

    q_token_type_ids = tf.keras.Input(shape=(Config.MAX_LENGTH), dtype=tf.int32)
    a_token_type_ids = tf.keras.Input(shape=(Config.MAX_LENGTH), dtype=tf.int32)

    config = transformers.BertConfig.from_pretrained(
        pretrained_model_name_or_path=Config.BERT_PATH + os.path.sep + "bert-base-uncased-config.json")
    bert = transformers.TFBertModel.from_pretrained(
        pretrained_model_name_or_path=Config.BERT_PATH + os.path.sep + "tf_model.bin", config=config)
    tmp = bert(q_ids, atttention_mask=q_mask, token_type_ids=q_token_type_ids)
    pass



if __name__ == '__main__':
    model()
