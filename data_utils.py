import random
from itertools import izip
import re
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.python.framework import constant_op
import numpy as np

_PAD = b"PAD"
_GO = b"GO"
_EOS = b"EOS"
_UNK = b"UNK"
_START_VOCAB = [_PAD, _EOS, _GO, _UNK]

PAD_ID = 0
EOS_ID = 1
GO_ID = 2
UNK_ID = 3


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

    def _gen_batched_data(self,
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

        batched_data = {'posts:0': np.array(posts),
                        'responses': np.array(responses),
                        'posts_length': posts_length,
                        'responses_length': responses_length}
        return batched_data

    def _next_batch(self,
                    data,
                    batch_size):
        selected_data = [random.choice(data) for i in range(batch_size)]
        batched_data = self._gen_batched_data(selected_data)
        return batched_data

    def batchify(self,
                 data,
                 batch_size):
        for i in range(len(data)//batch_size):
            yield self._next_batch(data, batch_size)


if __name__ == '__main__':
    h_dict = {}
    h_dict['data_dir'] = 'data'
    h_dict['splited'] = True
    h_dict['vocab_size'] = 4000
    h_dict['data_from'] = 'MSCOCO.p1.test'
    h_dict['data_to'] = 'MSCOCO.p2.test'
    h_dict['train_size'] = 10000
    h_dict['dev_size'] = 5000
    h_dict['test_size'] = 5000
    dp = data_process(h_dict)
    train_data = dp.load_train_data()
    print len(train_data)



    vocab = dp.build_vocab(train_data)
    symbol2index = MutableHashTable(
        key_dtype=tf.string,
        value_dtype=tf.int64,
        default_value=UNK_ID,
        shared_name="in_table",
        name="in_table",
        checkpoint=True)
    print vocab[:10]
    with tf.Session() as sess:
        op_in = symbol2index.insert(constant_op.constant(vocab),
                                    constant_op.constant(range(h_dict['vocab_size']), dtype=tf.int64))

        posts = tf.placeholder(tf.string, (None, None), 'posts')  # batch*len
        look = symbol2index.lookup(posts)
        print sess.run(op_in)
        cnt = 0
        # print sess.run(posts,{posts:train_data[0]['post']})
        for batch in dp.batchify(train_data,5):
            res = sess.run(posts,{batch.keys()[2]:batch.values()[2]})
            print res[1]
            res = sess.run(look, {batch.keys()[2]: batch.values()[2]})
            print res[1]
            break

