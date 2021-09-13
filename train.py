# -*- coding: UTF-8 -*-
import time
import traceback
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataloader import KeyphraseDataLoader
from evaluation import KeyphraseEvaluator
from utils import PAD_WORD, get_basename
import argparse
from collections import OrderedDict
from munch import Munch
from pysenal import write_json, read_json, get_logger
from utils import load_vocab
from model import CopyRNN
from dataloader import TOKENS, TARGET
from predict import CopyRnnPredictor


class CopyRnnTrainer:

    def __init__(self):
        torch.manual_seed(0)
        torch.autograd.set_detect_anomaly(True)
        self.args = self.parse_args()
        if self.args.train_from:
            self.dest_dir = os.path.dirname(self.args.train_from) + '/'
        else:
            timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
            self.dest_dir = os.path.join(self.args.dest_base_dir, self.args.exp_name + '-' + timemark) + '/'
            os.mkdir(self.dest_dir)

        # 日志输出
        self.logger = get_logger('train')
        fh = logging.FileHandler(os.path.join(self.dest_dir, self.args.logfile))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        self.logger.addHandler(fh)


        self.vocab2id = load_vocab(self.args.vocab_path, self.args.vocab_size)

        self.model = self.load_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        if self.args.train_parallel:  # 是否需要并行处理
            self.model = nn.DataParallel(self.model)

        self.loss_func = nn.NLLLoss(ignore_index=self.vocab2id[PAD_WORD])

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   self.args.schedule_step,
                                                   self.args.schedule_gamma)

        self.train_loader = KeyphraseDataLoader(data_source=self.args.train_filename,
                                                vocab2id=self.vocab2id,
                                                mode='train',
                                                args=self.args)


        if not self.args.tensorboard_dir:
            tensorboard_dir = self.dest_dir + 'logs/'
            self.logger.debug("tensorboard_dir不存在，我们令其为" + str(tensorboard_dir))
        else:
            tensorboard_dir = self.args.tensorboard_dir
            self.logger.debug("tensorboard_dir存在，是" + str(self.args.tensorboard_dir))
        self.writer = SummaryWriter(tensorboard_dir)

        # 评价指标
        self.eval_topn = (5, 10)
        self.macro_evaluator = KeyphraseEvaluator(self.eval_topn, 'macro',
                                                  self.args.token_field, self.args.keyphrase_field)
        self.micro_evaluator = KeyphraseEvaluator(self.eval_topn, 'micro',
                                                  self.args.token_field, self.args.keyphrase_field)
        self.best_f1 = None
        self.best_step = 0
        self.not_update_count = 0

    def load_model(self):
        self.logger.debug("load_model")
        if not self.args.train_from:
            model = CopyRNN(self.args, self.vocab2id)
        else:
            model_path = self.args.train_from
            config_path = os.path.join(os.path.dirname(model_path),
                                       get_basename(model_path) + '.json')

            old_config = read_json(config_path)
            old_config['train_from'] = model_path
            old_config['step'] = int(model_path.rsplit('_', 1)[-1].split('.')[0])
            self.args = Munch(old_config)
            self.vocab2id = load_vocab(self.args.vocab_path, self.args.vocab_size)

            model = CopyRNN(self.args, self.vocab2id)

            if torch.cuda.is_available():
                model_path = model_path[:-4] + 'model'
                checkpoint = torch.load(model_path)
            else:
                model_path = model_path[:-4] + 'model'
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

            state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    k = k[7:]
                state_dict[k] = v
            model.load_state_dict(state_dict)
        return model

    def train(self):
        try:
            self.train_func()
        except KeyboardInterrupt:
            self.logger.info('you terminate the train logic')
        except Exception:
            self.logger.error('exception occurred')
            err_stack = traceback.format_exc()
            self.logger.error(err_stack)
        finally:
            # terminate the loader processes
            del self.train_loader

    def train_func(self):
        self.logger.debug("train_func")
        step = 0
        is_stop = False
        if self.args.train_from:
            step = self.args.step
            self.logger.info('train from destination dir:{}'.format(self.dest_dir))
            self.logger.info('train from step {}'.format(step))
        else:
            self.logger.info('destination dir:{}'.format(self.dest_dir))

        for epoch in range(1, self.args.epochs + 1):
            self.logger.debug("现在是epoch"+str(epoch))
            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx % 100 == 0:
                    self.logger.debug("[epoch]"+str(epoch)+",[batch]"+str(batch_idx))
                try:
                    loss = self.train_batch(batch, step)
                except Exception as e:
                    err_stack = traceback.format_exc()
                    self.logger.error(err_stack)
                    loss = 0.0
                step += 1
                self.writer.add_scalar('loss', loss, step)
                del loss
                if step and step % self.args.save_model_step == 0:
                    torch.cuda.empty_cache()
                    self.evaluate_and_save_model(step, epoch)  # 见后面
                    torch.cuda.empty_cache()
                    if self.not_update_count >= self.args.early_stop_tolerance:
                        is_stop = True
                        break
            if is_stop:
                self.logger.info('best step {}'.format(self.best_step))
                break

    def train_batch(self, batch, step):
        self.model.train()
        loss = 0
        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            batch[TARGET] = batch[TARGET].cuda()
        targets = batch[TARGET]
        if self.args.auto_regressive:
            loss = self.get_auto_regressive_loss(batch, loss, targets)
        else:
            loss = self.get_one_pass_loss(batch, targets)

        loss.backward()

        # clip norm, this is very import for avoiding nan gradient and misconvergence
        if self.args.max_grad:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.max_grad)
        if self.args.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)

        self.optimizer.step()

        if self.args.schedule_lr and step <= self.args.schedule_step:
            self.scheduler.step()
        return loss

    def get_auto_regressive_loss(self, batch, loss, targets):

        batch_size = len(batch[TOKENS])
        encoder_output = None
        decoder_state = torch.zeros(batch_size, self.args.target_hidden_size)
        hidden_state = None
        for target_index in range(self.args.max_target_len):
            if target_index == 0:  # bos
                prev_output_tokens = targets[:, target_index].unsqueeze(1)
            else:
                if self.args.teacher_forcing:
                    prev_output_tokens = targets[:, target_index].unsqueeze(1)
                else:
                    best_probs, prev_output_tokens = torch.topk(decoder_prob, 1, 1)
            prev_output_tokens = prev_output_tokens.clone()
            output = self.model(src_dict=batch,
                                prev_output_tokens=prev_output_tokens,
                                encoder_output_dict=encoder_output,
                                prev_decoder_state=decoder_state,
                                prev_hidden_state=hidden_state)
            decoder_prob, encoder_output, decoder_state, hidden_state = output
            true_indices = targets[:, target_index + 1].clone()
            loss += self.loss_func(decoder_prob, true_indices)
        loss /= self.args.max_target_len
        return loss

    def get_one_pass_loss(self, batch, targets):
        batch_size = len(batch)
        encoder_output = None
        decoder_state = torch.zeros(batch_size, self.args.target_hidden_size)
        hidden_state = None
        prev_output_tokens = None  # 初始化第一个词
        output = self.model(src_dict=batch,
                            prev_output_tokens=prev_output_tokens,
                            encoder_output_dict=encoder_output,
                            prev_decoder_state=decoder_state,
                            prev_hidden_state=hidden_state)
        decoder_prob, encoder_output, decoder_state, hidden_state = output
        vocab_size = decoder_prob.size(-1)
        decoder_prob = decoder_prob.view(-1, vocab_size)
        loss = self.loss_func(decoder_prob, targets[:, 1:].flatten())
        return loss

    def evaluate_stage(self, step, stage, predict_callback):
        if stage == 'valid':
            src_filename = self.args.valid_filename
        elif stage == 'test':
            src_filename = self.args.test_filename
        else:
            raise ValueError('stage name error, must be in `valid` and `test`')

        src_filename_basename = get_basename(src_filename)
        pred_filename = self.dest_dir + src_filename_basename
        pred_filename += '.batch_{}.pred.jsonl'.format(step)

        torch.cuda.empty_cache()
        predict_callback()
        torch.cuda.empty_cache()

        macro_all_ret = self.macro_evaluator.evaluate(pred_filename)
        macro_present_ret = self.macro_evaluator.evaluate(pred_filename, 'present')
        macro_absent_ret = self.macro_evaluator.evaluate(pred_filename, 'absent')

        for n, counter in macro_all_ret.items():
            for k, v in counter.items():
                name = '{}/macro_{}@{}'.format(stage, k, n)
                self.writer.add_scalar(name, v, step)
        for n in self.eval_topn:
            name = 'present/{} macro_f1@{}'.format(stage, n)
            self.writer.add_scalar(name, macro_present_ret[n]['f1'], step)
        for n in self.eval_topn:
            absent_f1_name = 'absent/{} macro_f1@{}'.format(stage, n)
            self.writer.add_scalar(absent_f1_name, macro_absent_ret[n]['f1'], step)
            absent_recall_name = 'absent/{} macro_recall@{}'.format(stage, n)
            self.writer.add_scalar(absent_recall_name, macro_absent_ret[n]['recall'], step)

        statistics = {'{}_macro'.format(stage): macro_all_ret,
                      '{}_macro_present'.format(stage): macro_present_ret,
                      '{}_macro_absent'.format(stage): macro_absent_ret}
        return statistics

    def evaluate_and_save_model(self, step, epoch):
        valid_f1 = self.evaluate(step)
        if self.best_f1 is None:
            self.best_f1 = valid_f1
            self.best_step = step
        elif valid_f1 >= self.best_f1:
            self.best_f1 = valid_f1
            self.not_update_count = 0
            self.best_step = step
        else:
            self.not_update_count += 1

        exp_name = self.args.exp_name
        model_basename = self.dest_dir + '{}_epoch_{}_batch_{}'.format(exp_name, epoch, step)
        torch.save(self.model.state_dict(), model_basename + '.model')
        write_json(model_basename + '.json', vars(self.args))
        score_msg_tmpl = 'best score: step {} macro f1@{} {:.4f}'
        self.logger.info(score_msg_tmpl.format(self.best_step, self.eval_topn[-1], self.best_f1))
        self.logger.info('epoch {} step {}, model saved'.format(epoch, step))

    def evaluate(self, step):
        predictor = CopyRnnPredictor(model_info={'model': self.model, 'config': self.args},
                                     vocab_info=self.vocab2id,
                                     beam_size=self.args.beam_size,
                                     max_target_len=self.args.max_target_len,
                                     max_src_length=self.args.max_src_len)

        def pred_callback(stage):
            if stage == 'valid':
                src_filename = self.args.valid_filename
                dest_filename = self.dest_dir + get_basename(self.args.valid_filename)
            elif stage == 'test':
                src_filename = self.args.test_filename
                dest_filename = self.dest_dir + get_basename(self.args.test_filename)
            else:
                raise ValueError('stage name error, must be in `valid` and `test`')
            dest_filename += '.batch_{}.pred.jsonl'.format(step)

            def predict_func():
                predictor.eval_predict(src_filename=src_filename,
                                       dest_filename=dest_filename,
                                       args=self.args,
                                       model=self.model,
                                       remove_existed=True)

            return predict_func

        valid_statistics = self.evaluate_stage(step, 'valid', pred_callback('valid'))
        test_statistics = self.evaluate_stage(step, 'test', pred_callback('test'))
        total_statistics = {**valid_statistics, **test_statistics}

        eval_filename = self.dest_dir + self.args.exp_name + '.batch_{}.eval.json'.format(step)
        write_json(eval_filename, total_statistics)
        return valid_statistics['valid_macro'][self.eval_topn[-1]]['f1']



    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        # train and evaluation parameter
        parser.add_argument("-exp_name", required=True, type=str, help='')
        parser.add_argument("-train_filename", required=True, type=str, help='')
        parser.add_argument("-valid_filename", required=True, type=str, help='')
        parser.add_argument("-test_filename", required=True, type=str, help='')
        parser.add_argument("-dest_base_dir", required=True, type=str, help='')
        parser.add_argument("-vocab_path", required=True, type=str, help='')
        parser.add_argument("-vocab_size", type=int, default=500000, help='')
        parser.add_argument("-train_from", default='', type=str, help='')
        parser.add_argument("-token_field", default='tokens', type=str, help='')
        parser.add_argument("-keyphrase_field", default='keyphrases', type=str, help='')
        # add by sj
        parser.add_argument("-given_keyphrases_field", default='given_keyword_tokens', type=str, help='')

        parser.add_argument("-auto_regressive", action='store_true', help='')
        parser.add_argument("-epochs", type=int, default=10, help='')
        parser.add_argument("-batch_size", type=int, default=64, help='')
        parser.add_argument("-learning_rate", type=float, default=1e-4, help='')
        parser.add_argument("-eval_batch_size", type=int, default=50, help='')
        parser.add_argument("-dropout", type=float, default=0.0, help='')
        parser.add_argument("-grad_norm", type=float, default=0.0, help='')
        parser.add_argument("-max_grad", type=float, default=5.0, help='')
        parser.add_argument("-shuffle", action='store_true', help='')
        parser.add_argument("-teacher_forcing", action='store_true', help='')
        parser.add_argument("-beam_size", type=float, default=50, help='')
        parser.add_argument('-tensorboard_dir', type=str, default='', help='')
        parser.add_argument('-logfile', type=str, default='train_log.log', help='')
        parser.add_argument('-save_model_step', type=int, default=5000, help='')
        parser.add_argument('-early_stop_tolerance', type=int, default=100, help='')
        parser.add_argument('-train_parallel', action='store_true', help='')
        parser.add_argument('-schedule_lr', action='store_true', help='')
        parser.add_argument('-schedule_step', type=int, default=10000, help='')
        parser.add_argument('-schedule_gamma', type=float, default=0.1, help='')
        parser.add_argument('-processed', action='store_true', help='')
        parser.add_argument('-prefetch', action='store_true', help='')
        parser.add_argument('-backend', type=str, default='torch', help='')
        parser.add_argument('-lazy_loading', action='store_true', help='')
        parser.add_argument('-fix_batch_size', action='store_true', help='')

        # model specific parameter
        parser.add_argument("-embed_size", type=int, default=200, help='')
        parser.add_argument("-max_oov_count", type=int, default=100, help='')
        parser.add_argument("-max_src_len", type=int, default=1500, help='')
        parser.add_argument("-max_target_len", type=int, default=8, help='')
        parser.add_argument("-src_hidden_size", type=int, default=100, help='')
        parser.add_argument("-target_hidden_size", type=int, default=100, help='')
        parser.add_argument('-src_num_layers', type=int, default=1, help='')
        parser.add_argument('-target_num_layers', type=int, default=1, help='')
        parser.add_argument("-attention_mode", type=str, default='general',
                            choices=['general', 'dot', 'concat'], help='')
        parser.add_argument("-bidirectional", action='store_true', help='')
        parser.add_argument("-copy_net", action='store_true', help='')
        parser.add_argument("-input_feeding", action='store_true', help='')

        args = parser.parse_args(args)
        return args





if __name__ == '__main__':
    CopyRnnTrainer().train()
