# -*- coding: UTF-8 -*-
import json
from collections import namedtuple, OrderedDict
import numpy as np
import torch
from pysenal import read_lines_lazy, read_file
import re
import os

num_regex = re.compile(r'\d+([.]\d+)?')
char_regex = re.compile(r'[_\-—<>{,(?\\.\'%]|\d+([.]\d+)?', re.IGNORECASE)
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
DIGIT_WORD = '<digit>'
SEP_WORD = '<sep>'


def load_vocab(src_filename, vocab_size=None):
    """
    用src_filename里的词扩充词典
    :param src_filename: 源文件的路径
    :param vocab_size: 字典的大小
    :return:
    """
    vocab2id = {}
    for word in read_lines_lazy(src_filename):

        if word not in vocab2id:

            vocab2id[word] = len(vocab2id)

        if vocab_size and len(vocab2id) >= vocab_size:
            break

    if PAD_WORD not in vocab2id:
        raise ValueError('padding char is not in vocab')
    if UNK_WORD not in vocab2id:
        raise ValueError('unk char is not in vocab')
    if BOS_WORD not in vocab2id:
        raise ValueError('begin of sentence char is not in vocab')
    if EOS_WORD not in vocab2id:
        raise ValueError('end of sentence char is not in vocab')
    if DIGIT_WORD not in vocab2id:
        raise ValueError('digit char is not in vocab')
    if SEP_WORD not in vocab2id:
        raise ValueError('separator char is not in vocab')
    return vocab2id


def token_char_tokenize(text):
    text = char_regex.sub(r' \g<0> ', text)

    tokens = num_regex.sub(DIGIT_WORD, text).split()

    chars = []
    for token in tokens:

        if token == DIGIT_WORD:

            chars.append(token)
        else:

            chars.extend(list(token))
    return chars


def get_basename(filename):

    return os.path.splitext(os.path.basename(filename))[0]


def __check_eval_mode(eval_mode):
    if eval_mode not in {'all', 'present', 'absent'}:
        raise ValueError('evaluation mode must be in `all`, `present` and `absent`')


def load_config(model_info):
    if 'config' not in model_info:
        if isinstance(model_info['model'], str):
            config_path = os.path.splitext(model_info['model'])[0] + '.json'
        else:
            raise ValueError('config path is not assigned')
    else:
        config_info = model_info['config']
        if isinstance(config_info, str):
            config_path = config_info
        else:
            return config_info

    config = json.loads(read_file(config_path),
                        object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return config





def list2tensor(lst):
    return torch.as_tensor(np.array(lst, dtype=np.long)).long()
