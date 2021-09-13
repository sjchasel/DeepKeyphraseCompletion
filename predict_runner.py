# -*- coding: UTF-8 -*-
import argparse
from munch import Munch
from predict import CopyRnnPredictor
import logging
import time
from pysenal import read_json

timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='log/predict_runner-'+timemark+'.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

class PredictRunner:

    def __init__(self):
        self.args = self.parse_args()
        self.predictor = CopyRnnPredictor(model_info=self.args.model_path,
                                          vocab_info=self.args.vocab_path,
                                          beam_size=self.args.beam_size,
                                          max_src_length=self.args.max_src_len,
                                          max_target_len=self.args.max_target_len)
        self.config = Munch({**self.predictor.config, 'batch_size': self.args.batch_size})

    def predict(self):
        self.predictor.eval_predict(self.args.src_filename, self.args.dest_filename,
                                    args=self.config)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-src_filename', type=str, help='')
        parser.add_argument('-mode_path', type=str,default='data/kp20k/copyrnn_kp20k_basic-20210825/copyrnn_kp20k_basic_epoch_2_batch_115000.model', help='')
        parser.add_argument('-vocab_path', type=str, help='')
        parser.add_argument('-batch_size', type=int, default=10, help='')
        parser.add_argument('-beam_size', type=int, default=200, help='')
        parser.add_argument('-max_src_len', type=int, default=1500, help='')
        parser.add_argument('-max_target_len', type=int, default=8, help='')
        args = parser.parse_args()
        return args

    
if __name__ == '__main__':
    # Your model path
    model_path = 'data/kp20k/copyrnn_kp20k_basic-20210825/copyrnn_kp20k_basic_epoch_2_batch_115000.model'
    # model_path = 'data/kp20k/copyrnn_rnn/copyrnn_kp20k_basic_epoch_5_batch_55000.model'
    # your vocab path
    vocab_path = 'data/vocab_kp20k.txt'
    keyword_generator = CopyRnnPredictor({'model': model_path},
                                         vocab_info=vocab_path,
                                         beam_size=50,
                                         max_target_len=8,
                                         max_src_length=1500)

    from munch import Munch
    
    per = ['10','20','30','40','50']
    datasets = ['krapivin','nus','semeval','kp20k']
    config = read_json('data/kp20k/copyrnn_kp20k_basic-20210825/copyrnn_kp20k_basic_epoch_2_batch_115000.json')
    for pp in per:
        for data in datasets:
            src_filename = 'data/jsonl/'+data+'/'+data+'.test'+pp+'.jsonl'
            dest_filename = 'data/jsonl/'+data+'/'+data+'_pred'+pp+'.jsonl'
            keyword_generator.eval_predict(src_filename, dest_filename, args=Munch(config))

