import random
from itertools import izip
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op
from seq2seq import seq2seq, _START_VOCAB, _PAD, _EOS


class data_process(object):
    def __init__(self,
                 hparams_dict):
        self.data_dir = hparams_dict['data_dir']
        self.train_from = os.path.join(self.data_dir, 'train_from')
        self.train_to = os.path.join(self.data_dir, 'train_to')
        self.dev_from = os.path.join(self.data_dir, 'dev_from')
        self.dev_to = os.path.join(self.data_dir, 'dev_to')
        self.test_from = os.path.join(self.data_dir, 'test_from')
        self.test_to = os.path.join(self.data_dir, 'test_to')
        if not hparams_dict['splited']:
            self.data_from = os.path.join(self.data_dir,hparams_dict['data_from'])
            self.data_to = os.path.join(self.data_dir,hparams_dict['data_to'])
            self._split(hparams_dict['train_size'], hparams_dict['dev_size'], hparams_dict['test_size'])
        self.vocab_size = hparams_dict['vocab_size']


    def _split(self, train_size, dev_size, test_size):
        total_size = train_size+dev_size+test_size
        sel = random.sample(range(total_size), total_size)
        dev_dict = {}.fromkeys(sel[:dev_size])
        test_dict = {}.fromkeys(sel[-test_size:])
        train_from = open(self.train_from,'wb')
        train_to = open(self.train_to,'wb')
        dev_from = open(self.dev_from,'wb')
        dev_to = open(self.dev_to,'wb')
        test_from = open(self.test_from,'wb')
        test_to = open(self.test_to,'wb')

        with open(self.data_from) as ff, open(self.data_to) as ft:
            cntline = 0
            for post, resp in izip(ff,ft):
                if cntline in dev_dict:
                    dev_from.write(post)
                    dev_to.write(resp)
                elif cntline in test_dict:
                    test_from.write(post)
                    test_to.write(resp)
                else:
                    train_from.write(post)
                    train_to.write(resp)
                cntline+=1

        train_from.close()
        train_to.close()
        dev_from.close()
        dev_to.close()
        test_from.close()
        test_to.close()


    def build_vocab(self,
                    data):
        print("Creating vocabulary...")
        vocab = {}
        for i, pair in enumerate(data):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            for token in pair['post'] + pair['response']:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > self.vocab_size:
            vocab_list = vocab_list[:self.vocab_size]
        return vocab_list



    def load_train_data(self):
        return self.load_data(self.train_from,self.train_to)

    def load_dev_data(self):
        return self.load_data(self.dev_from, self.dev_to)

    def load_test_data(self):
        return self.load_data(self.test_from, self.test_to)

    def load_data(self,
                  post_f,
                  resp_f):
        f1 = open(post_f)
        f2 = open(resp_f)
        post = [line.strip().split() for line in f1.readlines()]
        response = [line.strip().split() for line in f2.readlines()]
        data = []
        for p, r in zip(post, response):
            data.append({'post': p, 'response': r})
        f1.close()
        f2.close()
        return data

    def gen_batched_data(self,
                         data):
        encoder_len = max([len(item['post']) for item in data]) + 1
        decoder_len = max([len(item['response']) for item in data]) + 1

        posts, responses, posts_length, responses_length = [], [], [], []

        def padding(sent, l):
            return sent + [_EOS] + [_PAD] * (l - len(sent) - 1)

        for item in data:
            posts.append(padding(item['post'], encoder_len))
            responses.append(padding(item['response'], decoder_len))
            posts_length.append(len(item['post']) + 1)
            responses_length.append(len(item['response']) + 1)

        batched_data = {'post:0': np.array(posts),
                        'response:0': np.array(responses),
                        'post_len:0': posts_length,
                        'response_len:0': responses_length}
        return batched_data

    def batchify(self,
                 data,
                 batch_size):
        for i in range(len(data)//batch_size):
            selected_data = [random.choice(data) for _ in range(batch_size)]
            batched_data = self.gen_batched_data(selected_data)
            yield batched_data

    def infer(self,
              s2s,
              sess):
        while True:
            infer_data = {}
            infer_data['post'] = raw_input('post > ').strip().split()
            infer_data['response'] = ''.strip().split()
            infer_data = [infer_data]
            for batch in self.batchify(infer_data, 1):
                res = sess.run([s2s.inference, s2s.beam_out], batch)
                print 'inference > ' + ' '.join(res[0][0])
                print 'beam > ' + ' '.join(res[1][0, :, 0])


if __name__ == '__main__':
    h_dict = {}
    h_dict['batch_size'] = 100
    h_dict['embed_size'] = 128
    h_dict['num_units'] = 50
    h_dict['num_layers'] = 1
    h_dict['beam_width'] = 5
    h_dict['use_lstm'] = True
    h_dict['attn_mode'] = 'Luong'
    h_dict['share_emb'] = True
    h_dict['lr'] = 0.001
    h_dict['data_dir'] = 'data'
    h_dict['splited'] = False
    h_dict['vocab_size'] = 4000
    h_dict['data_from'] = 'MSCOCO.p1.test'
    h_dict['data_to'] = 'MSCOCO.p2.test'
    h_dict['train_size'] = 10000
    h_dict['dev_size'] = 5000
    h_dict['test_size'] = 5000
    h_dict['keep_prob'] = 0.8
    h_dict['max_epoch'] = 2
    h_dict['is_train'] = True
    h_dict['train_dir'] = 'train'
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # print h_dict
    dp = data_process(h_dict)
    train_data = dp.load_train_data()
    valid_data = dp.load_dev_data()
    test_data = dp.load_test_data()
    # print len(train_data)
    vocab = dp.build_vocab(train_data)

    s2s = seq2seq(h_dict, vocab)
    with tf.Session() as sess:
        if h_dict['is_train']:
            if tf.train.get_checkpoint_state(h_dict['train_dir']):
                print("Reading model parameters from %s" % h_dict['train_dir'])
                s2s.saver.restore(sess, tf.train.latest_checkpoint(h_dict['train_dir']))
            else:
                print("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())
                sess.run([s2s.op_in, s2s.op_out])

            while True:
                # Keep track of average train cost for this epoch
                train_cost = 0
                for batch in dp.batchify(train_data, h_dict['batch_size']):
                    # print batch
                    res = sess.run([s2s.train_op, s2s.train_loss,
                                    ], batch)
                    train_cost += res[1]

                train_cost /= h_dict['train_size'] / h_dict['batch_size']

                test_cost = 0
                for batch in dp.batchify(valid_data, h_dict['batch_size']):
                    if test_cost == 0:
                        # Test time
                        t = sess.run([s2s.train_loss,
                                      s2s.post_string,
                                      s2s.response_string,
                                      s2s.train_out,
                                      s2s.inference,
                                      s2s.beam_out,
                                      # s2s.error_rate
                                      ], batch)
                        # er = t[6]
                        print 'post: ', t[1][0]
                        print 'response: ', t[2][0]
                        print 'train_out: ', t[3][0]
                        print 'infer_out: ', t[4][0]
                        print 'test_out: ', t[5][0, :, 0]
                    else:
                        t = sess.run([s2s.train_loss], batch)
                    test_cost += t[0]
                test_cost /= h_dict['test_size'] / h_dict['batch_size']
                print 'test ppl: ', np.exp(test_cost)

                print("train loss:", train_cost, "test loss:", test_cost)
                s2s.saver.save(sess, "%s/model.ckpt" % h_dict['train_dir'])

        else:
            s2s.saver.restore(sess, tf.train.latest_checkpoint(h_dict['train_dir']))
            dp.infer(s2s, sess)
