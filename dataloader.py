# -*- coding: UTF-8 -*-
import random
import traceback
import sys
import numpy as np
import multiprocessing
from pysenal import read_jsonline_lazy, get_chunk, read_jsonline, get_logger
from utils import *
import logging
import time

# add by sj
GIVEN_TOKENS = 'given_keyword_tokens'
GIVEN_TOKENS_LENS = 'given_tokens_len'

TOKENS = 'tokens'
TOKENS_LENS = 'tokens_len'
TOKENS_OOV = 'tokens_with_oov'

OOV_COUNT = 'oov_count'
OOV_LIST = 'oov_list'

TARGET_LIST = 'targets'
TARGET = 'target'

RAW_BATCH = 'raw'

TRAIN_MODE = 'train'
EVAL_MODE = 'eval'
INFERENCE_MODE = 'inference'


timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='log/dataloader-' + timemark + '.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )


class ExceptionWrapper(object):
    """
    Wraps an exception plus traceback to communicate across threads
    """

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]  # 异常的类型
        self.exc_msg = "".join(traceback.format_exception(*exc_info))  # 异常的信息


class KeyphraseDataLoader():
    def __init__(self, data_source, vocab2id, mode, args):
        self.data_source = data_source
        self.vocab2id = vocab2id
        self.mode = mode
        self.args = args
        self.vocab_size = len(vocab2id)  # args.vocab_size
        self.batch_size = args.batch_size
        self.max_src_len = args.max_src_len
        self.max_oov_count = args.max_oov_count
        self.max_target_len = args.max_target_len
        self.fix_batch_size = args.fix_batch_size
        self.prefetch = args.prefetch
        self.lazy_loading = args.lazy_loading
        self.shuffle = args.shuffle
        self.token_field = args.token_field
        self.keyphrases_field = args.keyphrase_field
        self.is_inference = mode != TRAIN_MODE
        # add by sj
        self.given_keyphrases_field = args.given_keyphrases_field


    def collate_fn(self, item):

        tokens = item[self.token_field]
        given_tokens = item[self.given_keyphrases_field]
        if len(tokens) > self.max_src_len:
            tokens = tokens[:self.max_src_len]
        for given_keyphrase in given_tokens:
            if len(given_keyphrase) > self.max_target_len:
                given_keyphrase = given_keyphrase[:self.max_target_len]

        token_ids_with_oov = []
        token_ids = []
        oov_list = []
        given_token_ids = []

        for token in tokens:
            idx = self.vocab2id.get(token, self.vocab_size)
            if idx == self.vocab_size:
                token_ids.append(self.vocab2id[UNK_WORD])
                if token not in oov_list:
                    if len(oov_list) >= self.max_oov_count:
                        token_ids_with_oov.append(self.vocab_size + self.max_oov_count - 1)
                    else:
                        token_ids_with_oov.append(self.vocab_size + len(oov_list))
                        oov_list.append(token)
                else:
                    token_ids_with_oov.append(self.vocab_size + oov_list.index(token))
            else:
                token_ids.append(idx)
                token_ids_with_oov.append(idx)

        for keyword_tokens in given_tokens:
            keyword_token_ids = []
            for token in keyword_tokens:
                idx = self.vocab2id.get(token, self.vocab_size)
                if idx == self.vocab_size:
                    keyword_token_ids.append(self.vocab2id[UNK_WORD])
                    if token not in oov_list:
                        if len(oov_list) < self.max_oov_count:
                            oov_list.append(token)
                else:
                    keyword_token_ids.append(idx)
            given_token_ids.append(keyword_token_ids)


        final_item = {TOKENS: token_ids,
                      GIVEN_TOKENS: given_token_ids,
                      TOKENS_OOV: token_ids_with_oov,
                      OOV_COUNT: len(oov_list),
                      OOV_LIST: oov_list}

        if self.is_inference:
            final_item[RAW_BATCH] = item
        else:
            keyphrase = item['phrase']
            target_ids = [self.vocab2id[BOS_WORD]]
            for token in keyphrase:
                target_ids.append(self.vocab2id.get(token, self.vocab2id[UNK_WORD]))
            target_ids.append(self.vocab2id[EOS_WORD])
            final_item[TARGET] = target_ids[:self.max_target_len]
        return final_item

    def __iter__(self):
        return iter(KeyphraseDataIterator(self))


class KeyphraseDataIterator:

    def __init__(self, loader):
        self.loader = loader
        self.data_source = loader.data_source
        self.batch_size = loader.batch_size
        self.lazy_loading = loader.lazy_loading
        self.backend = loader.args.backend
        self.fix_batch_size = loader.fix_batch_size
        self.workers = []
        #self.num_workers = multiprocessing.cpu_count() // 2 or 1
        self.num_workers = 1
        self.token_field = loader.token_field
        self.keyphrases_field = loader.keyphrases_field
        self.given_keyphrases_field = loader.given_keyphrases_field

        if self.loader.mode == TRAIN_MODE:
            self.chunk_size = self.batch_size * 5
        else:
            self.chunk_size = self.batch_size

        self._data = self.load_data(self.chunk_size)
        self._batch_count_in_output_queue = 0
        self._redundant_batch = []
        self.worker_shutdown = False

        if self.loader.mode in {TRAIN_MODE, EVAL_MODE}:
            self.input_queue = multiprocessing.Queue(-1)
            self.output_queue = multiprocessing.Queue(-1)
            self.__prefetch()
            for _ in range(self.num_workers):
                worker = multiprocessing.Process(target=self._data_worker_loop)
                self.workers.append(worker)
            for worker in self.workers:
                worker.daemon = True
                worker.start()


    def load_data(self, chunk_size):
        if isinstance(self.data_source, str):
            if self.lazy_loading:
                data = read_jsonline_lazy(self.data_source)
            else:
                data = read_jsonline(self.data_source)
                if self.loader.shuffle:
                    random.shuffle(data)
                    random.shuffle(data)
                    random.shuffle(data)
        elif isinstance(self.data_source, list):
            data = iter(self.data_source)
        else:
            raise TypeError('input filename type is error')
        return get_chunk(data, chunk_size)

    def _data_worker_loop(self):
        while True:
            raw_batch = self.input_queue.get()
            if raw_batch is None:
                break
            batch = self.padding_batch(raw_batch)  # 见下一个函数
            logging.debug("padding后的raw_batch是batch是"+str(batch))
            self.output_queue.put(batch)  # 填充之后放进output队列

    def padding_batch(self, raw_batch):
        max_src_len = self.loader.max_src_len
        max_target_len = self.loader.max_target_len
        pad_id = self.loader.vocab2id[PAD_WORD]
        max_keyphrases_num = 5

        token_ids_list = []
        token_len_list = []
        token_oov_ids_list = []
        oov_len_list = []
        oov_list = []
        raw_item_list = []
        target_ids_list = []
        given_token_ids_list = []
        given_token_len_list = []

        for raw_item in raw_batch:
            if not self.loader.args.processed:
                item = self.loader.collate_fn(raw_item)
            else:
                item = raw_item

            token_len = len(item[TOKENS])
            token_len_list.append(token_len)
            token_ids = item[TOKENS] + [pad_id] * (max_src_len - token_len)
            token_ids_list.append(token_ids)
            token_oov_ids = item[TOKENS_OOV] + [pad_id] * (max_src_len - token_len)
            token_oov_ids_list.append(token_oov_ids)
            oov_len_list.append(item[OOV_COUNT])
            oov_list.append(item[OOV_LIST])
            given_token_len = [len(i) for i in item[GIVEN_TOKENS]]
            given_token_len_list.append(given_token_len)
            for i in item[GIVEN_TOKENS]:
                given_token_ids = []
                if len(i) > (max_target_len + 1):
                    i = i[:max_target_len + 1]
                else:
                    i = i + [pad_id] * (max_target_len + 1 - len(i))
                given_token_ids.append(i)
            given_token_ids_list.append(given_token_ids)

            if self.loader.is_inference:
                raw_item_list.append(raw_item)
            else:
                target_ids = item[TARGET] + [pad_id] * (max_target_len + 1 - len(item[TARGET]))
                target_ids_list.append(target_ids)
        logging.debug("token_len_list:"+str(token_len_list))
        token_ids_np = np.array(token_ids_list, dtype=np.long)
        token_len_np = np.array(token_len_list, dtype=np.long)
        token_oov_np = np.array(token_oov_ids_list, dtype=np.long)
        oov_len_np = np.array(oov_len_list, dtype=np.long)
        for batch in given_token_len_list:
            while len(batch) < 5:
                batch.append(0)
        given_token_ids_np = np.array(given_token_ids_list, dtype=np.long)
        logging.debug("given_token_ids_np的类型是" + str(type(given_token_ids_np)) + "，shape是" + str(given_token_ids_np.shape))


        logging.debug("token_ids_np的类型是" + str(type(token_ids_np)) + "，shape是" + str(token_ids_np.shape))
        logging.debug("token_oov_np的类型是" + str(type(token_oov_np)) + "，shape是" + str(token_oov_np.shape))
        logging.debug("token_len_np的类型是" + str(type(token_len_np)) + "，shape是" + str(token_len_np.shape))
        logging.debug("given_token_ids_np的类型是" + str(type(given_token_ids_np)) + "，shape是" + str(given_token_ids_np.shape))

        batch = {TOKENS: token_ids_np,
                 TOKENS_OOV: token_oov_np,
                 TOKENS_LENS: token_len_np,
                 GIVEN_TOKENS: given_token_ids_np,
                 OOV_COUNT: oov_len_np,
                 OOV_LIST: oov_list}

        if not self.loader.is_inference:
            batch[TARGET] = np.array(target_ids_list, dtype=np.long)
        else:
            batch[RAW_BATCH] = raw_item_list
        return batch

    def __prefetch(self):
        if self.loader.mode == TRAIN_MODE:
            item_chunk = next(self._data)
            if self.loader.shuffle:
                random.shuffle(item_chunk)
            batches, redundant_batch = self.get_batches(item_chunk, [])
            self._redundant_batch = redundant_batch
            for batch in batches:
                self.input_queue.put(batch)
                self._batch_count_in_output_queue += 1
        else:
            for _ in range(self.num_workers):
                try:
                    item_chunk = next(self._data)
                except StopIteration:
                    break
                self.input_queue.put(item_chunk)
                self._batch_count_in_output_queue += 1


    def __iter__(self):
        if self.loader.mode == TRAIN_MODE:
            yield from self.iter_train()
        elif self.loader.mode == EVAL_MODE:
            yield from self.iter_inference_parallel()
        else:
            yield from self.iter_inference()

    def iter_train(self):
        redundant_batch = self._redundant_batch
        for item_chunk in self._data:
            if self.loader.shuffle:
                random.shuffle(item_chunk)
            batches, redundant_batch = self.get_batches(item_chunk, redundant_batch)
            batch_idx = 0
            for idx in range(self._batch_count_in_output_queue):
                if batch_idx < len(batches):
                    self.input_queue.put(batches[batch_idx])
                    batch_idx += 1
                res = self.output_queue.get()
                yield self.batch2tensor(res)

            if batch_idx < len(batches):
                for batch in batches[batch_idx:]:
                    self.input_queue.put(batch)

            self._batch_count_in_output_queue = len(batches)
            logging.debug("_batch_count_in_output_queue为"+str(self._batch_count_in_output_queue))
        if redundant_batch:
            self.input_queue.put(redundant_batch)
            self._batch_count_in_output_queue += 1

        if self._batch_count_in_output_queue:
            for idx in range(self._batch_count_in_output_queue):
                res = self.output_queue.get()
                logging.debug("iter_train里output_queue去出来的是"+str(res))
                yield self.batch2tensor(res)

    def iter_inference_parallel(self):
        assert not self._redundant_batch
        assert self.workers
        for item_chunk in self._data:
            if self._batch_count_in_output_queue > 0:
                yield self.batch2tensor(self.output_queue.get())
                self.input_queue.put(item_chunk)
            else:
                self.input_queue.put(item_chunk)
                yield self.batch2tensor(self.output_queue.get())
        if self._batch_count_in_output_queue:
            for _ in range(self._batch_count_in_output_queue):
                yield self.batch2tensor(self.output_queue.get())

    def iter_inference(self):
        assert not self._batch_count_in_output_queue
        assert not self._redundant_batch
        for item_chunk in self._data:
            yield self.batch2tensor(self.padding_batch(item_chunk))

    def get_batches(self, item_chunk, batch):
        if self.loader.args.processed:
            return self.get_batches_processed(item_chunk, batch)
        else:
            return self.get_batches_raw(item_chunk, batch)

    def get_batches_processed(self, item_chunk, batch):
        batches = []
        for new_batch in get_chunk(batch + item_chunk, self.batch_size):
            batches.append(new_batch)
        return batches, []

    def get_batches_raw(self, item_chunk, batch):
        batches = []

        for item in item_chunk:
            if self.fix_batch_size:
                if batch and len(batch) > self.batch_size:
                    tail_count = len(batch) % self.batch_size
                    if tail_count:
                        batch_chunk = batch[:-tail_count]
                        batch = batch[-tail_count:]
                    else:
                        batch_chunk = batch
                        batch = []
                    for sliced_batch in get_chunk(batch_chunk, self.batch_size):
                        batches.append(sliced_batch)

                flatten_items = self.flatten_raw_item(item)
                batch.extend(flatten_items)
            else:
                if batch and len(batch) > self.batch_size:
                    for sliced_batch in get_chunk(batch, self.batch_size):
                        batches.append(sliced_batch)
                    batch = []
                flatten_items = self.flatten_raw_item(item)
                if batch and len(batch) + len(flatten_items) > self.batch_size:
                    batches.append(batch)
                    batch = flatten_items
                else:
                    batch.extend(flatten_items)
        return batches, batch

    def flatten_raw_item(self, item):
        flatten_items = []
        for phrase in item[self.loader.keyphrases_field]:
            flatten_items.append({self.token_field: item[self.token_field],
                                  self.given_keyphrases_field: item[self.given_keyphrases_field], 'phrase': phrase})
        return flatten_items

    def batch2tensor(self, batch):
        new_batch = {}
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                import torch
                new_batch[key] = torch.as_tensor(val)
            else:
                new_batch[key] = val
        return new_batch


    def _shutdown_workers(self):
        if not self.workers:
            return

        self.input_queue.close()
        self.output_queue.close()

        for worker in self.workers:
            worker.terminate()
        self.workers = []



