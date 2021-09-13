# -*- coding: UTF-8 -*-
import os
import re
import argparse
import string
from collections import Counter
from itertools import chain
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pysenal import get_chunk, read_jsonline_lazy, append_jsonlines, write_lines
from utils import (PAD_WORD, UNK_WORD, DIGIT_WORD, BOS_WORD, EOS_WORD, SEP_WORD)
import random
import logging
import math

logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='data/kp20k/preprocess.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )


class Kp20kPreprocessor(object):
    num_and_punc_regex = re.compile(r'[_\-—<>{,(?\\.\'%]|\d+([.]\d+)?', re.IGNORECASE)  # 会匹配数字和字符
    num_regex = re.compile(r'\d+([.]\d+)?')  # 会匹配数字

    def __init__(self, args):
        self.src_filename = args.src_filename
        self.dest_filename = args.dest_filename
        self.dest_vocab_path = args.dest_vocab_path
        self.vocab_size = args.vocab_size
        self.parallel_count = args.parallel_count
        self.is_src_lower = args.src_lower
        self.is_src_stem = args.src_stem
        self.is_target_lower = args.target_lower
        self.is_target_stem = args.target_stem
        self.stemmer = PorterStemmer()
        if os.path.exists(self.dest_filename):
            print('destination file existed,' + self.dest_filename + ' will be deleted!!!')
            os.remove(self.dest_filename)
        self.flag = True

    def build_vocab(self, tokens):
        vocab = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, DIGIT_WORD, SEP_WORD]
        vocab.extend(list(string.digits))

        token_counter = Counter(tokens).most_common(self.vocab_size)
        for token, count in token_counter:
            vocab.append(token)
            if len(vocab) >= self.vocab_size:
                break
        logging.debug("vocab构建完毕")
        return vocab

    def process(self):

        pool = Pool(self.parallel_count)

        tokens = []
        chunk_size = 100
        for item_chunk in get_chunk(read_jsonline_lazy(self.src_filename), chunk_size):
            processed_records = pool.map(self.tokenize_record, item_chunk)

            if self.dest_vocab_path:
                for record in processed_records:
                    tokens.extend(record['title_and_abstract_tokens'] + record['flatten_keyword_tokens'])
            for record in processed_records:
                record.pop('flatten_keyword_tokens')
            processed_records = self.keyphrase_given(processed_records)
            append_jsonlines(self.dest_filename, processed_records)

        if self.dest_vocab_path:
            vocab = self.build_vocab(tokens)
            write_lines(self.dest_vocab_path, vocab)

    def tokenize_record(self, record):
        abstract_tokens = self.tokenize(record['abstract'], self.is_src_lower, self.is_src_stem)
        title_tokens = self.tokenize(record['title'], self.is_src_lower, self.is_src_stem)
        keyword_token_list = []
        for keyword in record['keyword'].split(';'):
            keyword_token_list.append(self.tokenize(keyword, self.is_target_lower, self.is_target_stem))
        result = {
            'title_and_abstract_tokens': title_tokens + abstract_tokens,
            'keyword_tokens': keyword_token_list,
            'flatten_keyword_tokens': list(chain(*keyword_token_list))
        }
        return result

    def keyphrase_given(self, data, per=50):
        remove_list = []
        if self.flag:
            logging.debug("传入的data是" + str(data))
        for dic in data:
            keywords_num = len(dic['keyword_tokens'])  # 关键短语的个数
            pre_num = math.floor(keywords_num*per/100)
            if keywords_num == 1 or pre_num == 0:
                remove_list.append(dic)
            givens = []
            for i in range(pre_num):
                given = random.choice(dic['keyword_tokens'])
                dic['keyword_tokens'].remove(given)
                givens.append(given)
            dic['given_keyword_tokens'] = givens
#         """
#         处理生成的jsonl文件，随机提取出1-3个keyphrase作为预先给定的关键短语
#         :return:
#         """
#         remove_list = []
#         if self.flag:
#             logging.debug("传入的data是" + str(data))
#         for dic in data:
#             keywords_num = len(dic['keyword_tokens'])  # 关键短语的个数
#             if keywords_num == 1:
#                 remove_list.append(dic)
#             elif keywords_num == 2 or keywords_num == 3:
#                 given = random.choice(dic['keyword_tokens'])
#                 dic['keyword_tokens'].remove(given)
#                 dic['given_keyword_tokens'] = given
#             elif keywords_num == 4 or keywords_num == 5:
#                 givens = []
#                 for i in range(2):
#                     given = random.choice(dic['keyword_tokens'])
#                     dic['keyword_tokens'].remove(given)
#                     givens.append(given)
#                 dic['given_keyword_tokens'] = givens
#             elif 6 <= keywords_num <= 20:
#                 givens = []
#                 for i in range(3):
#                     given = random.choice(dic['keyword_tokens'])
#                     dic['keyword_tokens'].remove(given)
#                     givens.append(given)
#                 dic['given_keyword_tokens'] = givens
#             else:
#                 givens = []
#                 for i in range(5):
#                     given = random.choice(dic['keyword_tokens'])
#                     dic['keyword_tokens'].remove(given)
#                     givens.append(given)
#                 dic['given_keyword_tokens'] = givens
        for dicc in remove_list:
            data.remove(dicc)
        if self.flag:
            logging.debug("返回的data是" + str(data))
            self.flag = False
        return data

    def tokenize(self, text, is_lower, is_stem):

        text = self.num_and_punc_regex.sub(r' \g<0> ', text)
        tokens = word_tokenize(text)
        if is_lower:
            tokens = [token.lower() for token in tokens]
        if is_stem:
            tokens = [self.stemmer.stem(token) for token in tokens]
        for idx, token in enumerate(tokens):
            token = tokens[idx]
            if self.num_regex.fullmatch(token):
                tokens[idx] = DIGIT_WORD
        return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_filename', type=str, required=True,
                        help='input source kp20k file path')
    parser.add_argument('-dest_filename', type=str, required=True,
                        help='destination of processed file path')
    parser.add_argument('-dest_vocab_path', type=str,
                        help='')
    parser.add_argument('-vocab_size', type=int, default=50000,
                        help='')
    parser.add_argument('-parallel_count', type=int, default=10)
    parser.add_argument('-src_lower', action='store_true')
    parser.add_argument('-src_stem', action='store_true')
    parser.add_argument('-target_lower', action='store_true')
    parser.add_argument('-target_stem', action='store_true')

    args = parser.parse_args()
    processor = Kp20kPreprocessor(args)
    processor.process()


if __name__ == '__main__':
    main()
