# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from pysenal import get_logger
from dataloader import TOKENS, TOKENS_OOV, TOKENS_LENS, OOV_COUNT, TARGET, GIVEN_TOKENS  # add by sj
import logging
import time

timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
logger = get_logger('model')
fh = logging.FileHandler('log/model-' + timemark + '.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
logger.addHandler(fh)


class Attention0(nn.Module):
    """
    聚合关键短语向量的注意力机制
    """

    def __init__(self, input_dim, output_dim, score_mode='general'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.score_mode = score_mode
        if self.score_mode == 'general':
            self.attn = nn.Linear(self.input_dim, self.output_dim, bias=False)
        else:
            raise ValueError('attention score mode error')
        self.output_proj = nn.Linear(self.input_dim + self.output_dim, self.output_dim)

    def score(self, query, key, encoder_padding_mask):

        tgt_len = query.size(1)
        src_len = key.size(1)

        if self.score_mode == 'general':
            mat1 = self.attn(
                query)
            mat2 = key.permute(0, 2, 1)
            attn_weights = torch.bmm(mat1, mat2)
        attn_weights = torch.softmax(attn_weights,
                                     2)

        return attn_weights

    def forward(self, encoder_output, given_keyphrases, encoder_padding_mask):
        # query: encoder_output : B * max_src_len * hidden_size*2
        # key: given_keyphrases : B * keyphrase_num * embed_size
        attn_weights = self.score(encoder_output, given_keyphrases, encoder_padding_mask)
        context_embed = torch.bmm(attn_weights, given_keyphrases)
        # attn_weights:B * max_src_len * keyphrase_num
        # given_keyphrases: B * keyphrase_num * embed_size
        # context_embed: B * max_src_len * embed_size
        attn_outputs = torch.cat([context_embed, encoder_output], dim=2)  # 把h和c拼起来作为h（encoder_output）
        # attn_outputs : B * max_src_len * (2*hidden_size + embed_size)
        return attn_outputs, attn_weights


class CopyRnnEncoder(nn.Module):
    def __init__(self, vocab2id, embedding, hidden_size, bidirectional, dropout, args):
        super().__init__()
        embed_dim = embedding.embedding_dim
        self.embed_dim = embed_dim
        self.embedding = embedding  # nn.Embedding
        self.hidden_size = hidden_size  # default=100
        self.bidirectional = bidirectional
        self.num_layers = 1
        self.args = args
        self.src_hidden_size = self.args.src_hidden_size
        self.pad_idx = vocab2id[PAD_WORD]
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            batch_first=True)
        if bidirectional:
            bi = 2
        else:
            bi = 1
        self.attn_layer0 = Attention0(bi * hidden_size, embed_dim, args.attention_mode)

    def forward(self, src_dict):
        src_tokens = src_dict[TOKENS]
        src_lengths = src_dict[TOKENS_LENS]
        batch_size = len(src_tokens)
        src_embed = self.embedding(src_tokens)
        src_embed = F.dropout(src_embed, p=self.dropout, training=self.training)
        total_length = src_embed.size(1)
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(src_embed,
                                                             src_lengths.cpu(),
                                                             batch_first=True,
                                                             enforce_sorted=False)
        state_size = [self.num_layers, batch_size, self.hidden_size]
        if self.bidirectional:
            state_size[0] *= 2
        h0 = src_embed.new_zeros(state_size)
        hidden_states, final_hiddens = self.gru(packed_src_embed, h0)
        # final_hiddens\final_cells: 2 * 2/14/15/16 * 100
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(hidden_states,
                                                            padding_value=self.pad_idx,
                                                            batch_first=True,
                                                            total_length=total_length)

        encoder_padding_mask = src_tokens.eq(self.pad_idx)
        if self.bidirectional:
            final_hiddens = torch.cat((final_hiddens[0], final_hiddens[1]), dim=1).unsqueeze(0)
        given_tokens = src_dict[GIVEN_TOKENS]
        given_embed = self.embedding(given_tokens)
        given_embed = F.dropout(given_embed, p=self.dropout, training=self.training)
        given_embed = given_embed.mean(dim=1)  # B * max_keyphrase_num(5) * embed_size

        encoder_padding_mask2 = torch.full(given_tokens.shape, 1)
        attn_output, attn_weights = self.attn_layer0(hidden_states, given_embed, encoder_padding_mask2)
        # hidden_states : B * max_src_len * hidden_size*2
        # given_embed : B * keyphrase_num * embed_size
        # attn_weights : B * max_src_len * keyphrase_num
        # attn_output:  B * max_src_len * (2*hidden_size + embed_size)

        final_hiddens = attn_output[:, -1, :].unsqueeze(0)  # 1 * B * (2*hidden_size + embed_size)

        output = {'encoder_output': attn_output,  # B * max_src_len * (2*hidden_size + embed_size)
                  'encoder_padding_mask': encoder_padding_mask,
                  'encoder_hidden': final_hiddens}

        return output


class Attention(nn.Module):
    """
    implement attention mechanism
    """

    def __init__(self, input_dim, output_dim, score_mode='general'):
        super().__init__()
        self.input_dim = input_dim  # SH--->SH + embed_size
        self.output_dim = output_dim  # TH
        self.score_mode = score_mode
        self.attn = nn.Linear(self.output_dim, self.input_dim, bias=False)
        # TH * SH + embed_size
        self.output_proj = nn.Linear(self.input_dim + self.output_dim, self.output_dim)
        # SH+embed_size+TH * TH

    def score(self, query, key, encoder_padding_mask):
        """
        :param query: decoder_output B * TL * TH
        :param key: encoder_outputs : B * L * (SH + embed_size)
        :param encoder_padding_mask:
        :return:
        """
        tgt_len = query.size(1)
        src_len = key.size(1)
        attn_weights = torch.bmm(self.attn(query), key.permute(0, 2, 1))
        # self.attn(query):  B * TL * SH + embed_size
        # key.permute(0, 2, 1): B * (SH + embed_size) * L
        # attn_weights: B * TL * L

        # mask input padding to -Inf, they will be zero after softmax.
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1).repeat(1, tgt_len, 1)
            attn_weights.masked_fill_(encoder_padding_mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, 2)
        return attn_weights

    def forward(self, decoder_output, encoder_outputs, encoder_padding_mask):
        """
        :param decoder_output: B * TL * tgt_dim
        :param encoder_outputs: B x L x src_dim
        :param encoder_padding_mask:
        :return:
        """
        attn_weights = self.score(decoder_output, encoder_outputs, encoder_padding_mask)
        # B * TL * L
        context_embed = torch.bmm(attn_weights, encoder_outputs)
        # B * TL * L与 B * L * (SH + embed_size)-->B * TL * (SH + embed_size)
        attn_outputs = torch.tanh(self.output_proj(torch.cat([context_embed, decoder_output], dim=2)))
        # torch.cat([context_embed, decoder_output]: B * TL * (SH + embed_size + TH)
        # self.output_proj(...): B * TL * TH
        # attn_output: B * TL * TH
        return attn_outputs, attn_weights
        # attn_output: B * TL * TH
        # attn_weights: B * TL * L


class CopyRnnDecoder(nn.Module):
    def __init__(self, vocab2id, embedding, args):
        super().__init__()
        self.vocab2id = vocab2id
        vocab_size = embedding.num_embeddings  # 有几个词
        embed_dim = embedding.embedding_dim  # 每个词的维度是多少
        self.vocab_size = vocab_size
        self.embed_size = embed_dim
        self.embedding = embedding

        self.target_hidden_size = args.target_hidden_size  # decoder不是双向
        if args.bidirectional:
            self.src_hidden_size = args.src_hidden_size * 2
        else:
            self.src_hidden_size = args.src_hidden_size

        self.max_src_len = args.max_src_len
        self.max_oov_count = args.max_oov_count
        self.dropout = args.dropout
        self.pad_idx = vocab2id[PAD_WORD]
        self.is_copy = args.copy_net
        self.input_feeding = args.input_feeding
        self.auto_regressive = args.auto_regressive

        if not self.auto_regressive and self.input_feeding:
            raise ValueError('auto regressive must be used when input_feeding is on')

        decoder_input_size = embed_dim
        if args.input_feeding:
            decoder_input_size += self.src_hidden_size

        self.gru = nn.GRU(
            input_size=decoder_input_size,
            hidden_size=self.target_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.attn_layer = Attention(self.src_hidden_size + embed_dim, self.target_hidden_size, args.attention_mode)
        self.copy_proj = nn.Linear(self.src_hidden_size + embed_dim, self.target_hidden_size, bias=False)
        self.input_copy_proj = nn.Linear(self.src_hidden_size + embed_dim, self.target_hidden_size, bias=False)
        self.generate_proj = nn.Linear(self.target_hidden_size, self.vocab_size, bias=False)

    def forward(self, prev_output_tokens, encoder_output_dict, prev_context_state,
                prev_rnn_state, src_dict):

        if self.is_copy:
            if self.auto_regressive or not self.training:
                output = self.forward_copyrnn_auto_regressive(encoder_output_dict=encoder_output_dict,
                                                              prev_context_state=prev_context_state,
                                                              prev_output_tokens=prev_output_tokens,
                                                              prev_rnn_state=prev_rnn_state,
                                                              src_dict=src_dict)
            else:
                output = self.forward_copyrnn_one_pass(encoder_output_dict=encoder_output_dict,
                                                       src_dict=src_dict,
                                                       encoder_hidden_state=prev_rnn_state)

        return output

    def forward_copyrnn_auto_regressive(self,
                                        encoder_output_dict,
                                        prev_context_state,
                                        prev_output_tokens,
                                        prev_rnn_state,
                                        src_dict):

        src_tokens = src_dict[TOKENS]
        src_tokens_with_oov = src_dict[TOKENS_OOV]
        batch_size = len(src_tokens)

        prev_output_tokens = torch.as_tensor(prev_output_tokens, dtype=torch.int64)
        if torch.cuda.is_available():
            prev_output_tokens = prev_output_tokens.cuda()

        encoder_output = encoder_output_dict['encoder_output']
        encoder_output_mask = encoder_output_dict['encoder_padding_mask']
        # B x 1 x L
        copy_state = self.get_attn_read_input(encoder_output,
                                              prev_context_state,
                                              prev_output_tokens,
                                              src_tokens_with_oov)
        # copy state: beam_batch_size * 1* 2*hidden+embed_size

        # map copied oov tokens to OOV idx to avoid embedding lookup error
        prev_output_tokens[prev_output_tokens >= self.vocab_size] = self.vocab2id[UNK_WORD]
        src_embed = self.embedding(prev_output_tokens)
        # 我猜在auto regressive中是B * 1 * embed_dim

        if self.input_feeding:
            decoder_input = torch.cat([src_embed, copy_state], dim=2)
        else:
            decoder_input = src_embed
        decoder_input = F.dropout(decoder_input, p=self.dropout, training=self.training)
        rnn_output, rnn_state = self.gru(decoder_input, prev_rnn_state)
        # decoder_input: B * 1 * embed_dim
        # prev_rnn_state: 1 * B * tgt_hidden_size
        # rnn_output: B * 1 * tgt_hidden_size
        # rnn_state: 1 * B * tgt_hidden_size
        # attn_output is the final hidden state of decoder layer
        # attn_output B x 1 x TH
        attn_output, attn_weights = self.attn_layer(rnn_output, encoder_output, encoder_output_mask)
        # rnn_output: B * 1 * tgt_hidden_size
        # encoder_output: B * max_src_len * (2*hidden_size + embed_size)
        # attn_output: B * 1 * TH
        # attn_weights: B * 1 * max_src_len
        generate_logits = torch.exp(self.generate_proj(attn_output).squeeze(1))
        # self.generate_proj(attn_output): B * 1 * V
        # self.generate_proj(attn_output).squeeze(1): B * V
        # generate_logits： B * V

        # add 1e-10 to avoid -inf in torch.log
        generate_oov_logits = torch.zeros(batch_size, self.max_oov_count) + 1e-10
        if torch.cuda.is_available():
            generate_oov_logits = generate_oov_logits.cuda()
        generate_logits = torch.cat([generate_logits, generate_oov_logits], dim=1)
        # generate_logits: B * (V +OOV_V)
        copy_logits = self.get_copy_score(encoder_output,
                                          src_tokens_with_oov,
                                          attn_output,
                                          encoder_output_mask)
        # copy_logits: B * TL * (V + oov_V)

        # log softmax
        # !! important !!
        # must add the generative and copy logits after exp func , so tf.log_softmax can't be called
        # because it will add the generative and copy logits before exp func, then it's equal to multiply
        # the exp(generative) and exp(copy) result, not the sum of them.
        total_logit = generate_logits + copy_logits.squeeze(1)
        # total_logit: B * (V + OOV_V)
        total_prob = total_logit / torch.sum(total_logit, 1).unsqueeze(1)
        # total_prob：B * (V + OOV_V)
        total_prob = torch.log(total_prob)
        return total_prob, attn_output.squeeze(1), rnn_state
        # total_prob： B * (V + OOV_V)
        # attn_output.squeeze(1): B * TH
        # rnn_state: 1 * B * tgt_hidden_size

    def forward_copyrnn_one_pass(self, encoder_output_dict, encoder_hidden_state, src_dict):

        # TL
        dec_len = src_dict[TARGET].size(1) - 1
        src_tokens_with_oov = src_dict[TOKENS_OOV]
        batch_size = len(src_tokens_with_oov)

        encoder_output = encoder_output_dict[
            'encoder_output']  # B * max_src_len * (2*hidden_size（src_hidden_size） + embed_size)
        encoder_output_mask = encoder_output_dict['encoder_padding_mask']

        decoder_input = self.embedding(src_dict[TARGET][:, :-1])
        # decoder_input: B * TL * embed_size
        rnn_output, rnn_state = self.gru(decoder_input, encoder_hidden_state)
        # decoder_input: B * TL * embed_size
        # encoder_hidden_state：1 * B * TH
        attn_output, attn_weights = self.attn_layer(rnn_output, encoder_output, encoder_output_mask)
        # rnn_output: B * TL * hidden_size
        # encoder_output: B * max_src_len * (2*hidden_size + embed_size)

        # attn_outputs：B * TL * TH
        # attn_weights：B * TL * L

        generate_logits = torch.exp(self.generate_proj(attn_output))
        # attn_outputs：B * TL * TH
        # generate_logits : B * TL * V

        # add 1e-10 to avoid -inf in torch.log
        # oov词汇生成的概率：B * TL * oov_size
        generate_oov_logits = torch.zeros(batch_size, dec_len, self.max_oov_count) + 1e-10
        if torch.cuda.is_available():
            generate_oov_logits = generate_oov_logits.cuda()
        generate_logits = torch.cat([generate_logits, generate_oov_logits], dim=2)

        copy_logits = self.get_copy_score(encoder_output,
                                          src_tokens_with_oov,
                                          attn_output,
                                          encoder_output_mask)

        total_logit = generate_logits + copy_logits
        ## B * TL * (vocab_size + oov_size)
        total_prob = total_logit / torch.sum(total_logit, 2).unsqueeze(2)
        total_prob = torch.log(total_prob)
        return total_prob, attn_output, rnn_state
        # total_prob:  B * TL(max_tgt_len) * (vocab_size + oov_size)
        # attn_output: B * TL(max_tgt_len) * TH
        # rnn_state: final_hidden： 1 * B * TH


    def get_attn_read_input(self, encoder_output, prev_context_state,
                            prev_output_tokens, src_tokens_with_oov):

        # mask : B x L x 1
        mask_bool = torch.eq(prev_output_tokens.repeat(1, self.max_src_len),
                             src_tokens_with_oov).unsqueeze(2)
        mask = mask_bool.type_as(encoder_output)

        aggregate_weight = torch.tanh(self.input_copy_proj(torch.mul(mask, encoder_output)))

        no_zero_mask = ((mask != 0).sum(dim=1) != 0).repeat(1, self.max_src_len).unsqueeze(2)
        input_copy_logit_mask = no_zero_mask * mask_bool
        input_copy_logit = torch.bmm(aggregate_weight, prev_context_state.unsqueeze(2))
        input_copy_logit.masked_fill_(input_copy_logit_mask, float('-inf'))
        input_copy_weight = torch.softmax(input_copy_logit.squeeze(2), 1)
        # B x 1 x SH
        copy_state = torch.bmm(input_copy_weight.unsqueeze(1), encoder_output)
        return copy_state

    def get_copy_score(self, encoder_out, src_tokens_with_oov, decoder_output, encoder_output_mask):

        dec_len = decoder_output.size(1)  # TL（1）
        batch_size = len(encoder_out)

        copy_score_in_seq = torch.bmm(torch.tanh(self.copy_proj(encoder_out)),
                                      decoder_output.permute(0, 2, 1))

        copy_score_mask = encoder_output_mask.unsqueeze(2).repeat(1, 1, dec_len)
        # copy_score_mask: B * L * TL
        copy_score_in_seq.masked_fill_(copy_score_mask, float('-inf'))
        copy_score_in_seq = torch.exp(copy_score_in_seq)
        total_vocab_size = self.vocab_size + self.max_oov_count
        copy_score_in_vocab = torch.zeros(batch_size, total_vocab_size, dec_len)
        # B * (vocab_size + oov_size) * max_tgt_len
        if torch.cuda.is_available():
            copy_score_in_vocab = copy_score_in_vocab.cuda()
        token_ids = src_tokens_with_oov.unsqueeze(2).repeat(1, 1, dec_len)
        copy_score_in_vocab.scatter_add_(1, token_ids, copy_score_in_seq)
        copy_score_in_vocab = copy_score_in_vocab.permute(0, 2, 1)
        return copy_score_in_vocab  # B x TL x (vocab_size + oov_size)


class CopyRNN(nn.Module):

    def __init__(self, args, vocab2id):
        super().__init__()
        src_hidden_size = args.src_hidden_size
        target_hidden_size = args.target_hidden_size  # 100
        embed_size = args.embed_size
        embedding = nn.Embedding(len(vocab2id), embed_size, padding_idx=vocab2id[PAD_WORD])
        nn.init.uniform_(embedding.weight, -0.1, 0.1)

        self.encoder = CopyRnnEncoder(vocab2id=vocab2id,
                                      embedding=embedding,
                                      hidden_size=src_hidden_size,
                                      bidirectional=args.bidirectional,
                                      dropout=args.dropout,
                                      args=args)
        if args.bidirectional:
            decoder_src_hidden_size = 2 * src_hidden_size + embed_size  # 400
        else:
            decoder_src_hidden_size = src_hidden_size + embed_size
        self.decoder = CopyRnnDecoder(vocab2id=vocab2id, embedding=embedding, args=args)
        if decoder_src_hidden_size != target_hidden_size:
            self.encoder2decoder_state = nn.Linear(decoder_src_hidden_size, target_hidden_size)

    def forward(self, src_dict, prev_output_tokens, encoder_output_dict,
                prev_decoder_state, prev_hidden_state):
        if torch.cuda.is_available():
            src_dict[TOKENS] = src_dict[TOKENS].cuda()
            src_dict[TOKENS_LENS] = src_dict[TOKENS_LENS].cuda()
            src_dict[TOKENS_OOV] = src_dict[TOKENS_OOV].cuda()
            src_dict[OOV_COUNT] = src_dict[OOV_COUNT].cuda()
            src_dict[GIVEN_TOKENS] = src_dict[GIVEN_TOKENS].cuda()
            # src_dict[GIVEN_TOKENS_LENS] = src_dict[GIVEN_TOKENS_LENS].cuda()
            if prev_output_tokens is not None:
                prev_output_tokens = prev_output_tokens.cuda()
            prev_decoder_state = prev_decoder_state.cuda()
        if encoder_output_dict is None:
            encoder_output_dict = self.encoder(src_dict)
            prev_hidden_state = encoder_output_dict['encoder_hidden']
            # 1 * B * (2*hidden_size + embed_size)
            prev_hidden_state = self.encoder2decoder_state(prev_hidden_state)
            # 1 * B * TH

        decoder_prob, prev_decoder_state, prev_hidden_state = self.decoder(
            src_dict=src_dict,
            prev_output_tokens=prev_output_tokens,
            encoder_output_dict=encoder_output_dict,
            prev_context_state=prev_decoder_state,
            prev_rnn_state=prev_hidden_state)

        return decoder_prob, encoder_output_dict, prev_decoder_state, prev_hidden_state
