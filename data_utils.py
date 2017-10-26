import random
from itertools import izip
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op
import time
from seq2seq import seq2seq, _START_VOCAB, _PAD, _EOS

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_from", "MSCOCO.p1.train", "data_from")
tf.app.flags.DEFINE_string("data_to", "MSCOCO.p2.train", "data_to")
tf.app.flags.DEFINE_string("data_dir", "./data", "data_dir")
tf.app.flags.DEFINE_string("train_dir", "./train", "train_dir")
tf.app.flags.DEFINE_string("log_dir", "./log", "log_dir")
tf.app.flags.DEFINE_string("attn_mode", "Luong", "attn_mode")

tf.app.flags.DEFINE_boolean("use_lstm", True, "use_lstm")
tf.app.flags.DEFINE_boolean("share_emb", False, "share_emb")
tf.app.flags.DEFINE_boolean("split", True, "whether the data have been split in to train/dev/test")
tf.app.flags.DEFINE_boolean("is_train", True, "is_train")

tf.app.flags.DEFINE_integer("batch_size", 100, "batch_size")
tf.app.flags.DEFINE_integer("embed_size", 128, "embed_size")
tf.app.flags.DEFINE_integer("num_units", 512, "num_units")
tf.app.flags.DEFINE_integer("num_layers", 4, "num_layers")
tf.app.flags.DEFINE_integer("beam_width", 5, "beam_width")
tf.app.flags.DEFINE_integer("vocab_size", 40000, "vocab_size")
tf.app.flags.DEFINE_integer("train_size", 206902, "train_size")
tf.app.flags.DEFINE_integer("valid_size", 10000, "valid_size")
tf.app.flags.DEFINE_integer("test_size", 10000, "test_size")
tf.app.flags.DEFINE_integer("save_every_n_iteration", 1000, "save_every_n_iteration")

tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_float("keep_prob", 0.8, "keep_prob")


class data_process(object):
    def __init__(self,
                 tfFLAGS):
        self.data_dir = tfFLAGS.data_dir
        self.train_from = os.path.join(self.data_dir, 'train_from')
        self.train_to = os.path.join(self.data_dir, 'train_to')
        self.valid_from = os.path.join(self.data_dir, 'valid_from')
        self.valid_to = os.path.join(self.data_dir, 'valid_to')
        self.test_from = os.path.join(self.data_dir, 'test_from')
        self.test_to = os.path.join(self.data_dir, 'test_to')
        if not tfFLAGS.split:
            self.data_from = os.path.join(self.data_dir,tfFLAGS.data_from)
            self.data_to = os.path.join(self.data_dir,tfFLAGS.data_to)
            self._split(tfFLAGS.train_size, tfFLAGS.valid_size, tfFLAGS.test_size)
        self.vocab_size = tfFLAGS.vocab_size

    def _split(self, train_size, valid_size, test_size):
        total_size = train_size+valid_size+test_size
        sel = random.sample(range(total_size), total_size)
        valid_dict = {}.fromkeys(sel[:valid_size])
        test_dict = {}.fromkeys(sel[-test_size:])
        train_from = open(self.train_from,'wb')
        train_to = open(self.train_to,'wb')
        valid_from = open(self.valid_from,'wb')
        valid_to = open(self.valid_to,'wb')
        test_from = open(self.test_from,'wb')
        test_to = open(self.test_to,'wb')

        with open(self.data_from) as ff, open(self.data_to) as ft:
            cntline = 0
            for post, resp in izip(ff,ft):
                if cntline in valid_dict:
                    valid_from.write(post)
                    valid_to.write(resp)
                elif cntline in test_dict:
                    test_from.write(post)
                    test_to.write(resp)
                else:
                    train_from.write(post)
                    train_to.write(resp)
                cntline+=1

        train_from.close()
        train_to.close()
        valid_from.close()
        valid_to.close()
        test_from.close()
        test_to.close()
        print "split completed"

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

    def load_valid_data(self):
        return self.load_data(self.valid_from, self.valid_to)

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

        batched_data = {'post': np.array(posts),
                        'response': np.array(responses),
                        'post_len': posts_length,
                        'response_len': responses_length}
        return batched_data

    def train_batch(self, data, batch_size):
        selected_data = [random.choice(data) for _ in range(batch_size)]
        batched_data = self.gen_batched_data(selected_data)
        return batched_data

    def eval_batches(self, data, batch_size):
        for i in range(len(data)//batch_size):
            batched_data = self.gen_batched_data(data[i*batch_size:(i+1)*batch_size])
            yield batched_data

    def infer(self,
              s2s,
              sess):
        while True:
            infer_data = {}
            infer_data['post'] = raw_input('post > ').strip().split()
            infer_data['response'] = ''.strip().split()
            infer_data = [infer_data]
            batch = self.gen_batched_data(infer_data)
            res = sess.run([s2s.inference, s2s.beam_out], batch)
            print 'inference > ' + ' '.join(res[0][0])
            print 'beam > ' + ' '.join(res[1][0, :, 0])


def main(unused_argv):

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    dp = data_process(FLAGS)
    train_data = dp.load_train_data()
    valid_data = dp.load_valid_data()
    test_data = dp.load_test_data()

    s2s = seq2seq(FLAGS)
    with tf.Session() as sess:
        if FLAGS.is_train:
            if tf.train.get_checkpoint_state(FLAGS.train_dir):
                print("Reading model parameters from %s" % FLAGS.train_dir)
                s2s.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
            else:
                print("Created model with fresh parameters.")
                vocab = dp.build_vocab(train_data)
                s2s.initialize(sess, vocab=vocab)

            train_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'train'), sess.graph)
            valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'valid'))

            train_loss = 0
            time_step = 0
            while True:
                start_time = time.time()

                train_batch = dp.train_batch(train_data, FLAGS.batch_size)
                train_op, loss, global_step = s2s.step(sess, train_batch, is_train=True)
                train_loss += loss
                time_step += (time.time() - start_time)

                if global_step % FLAGS.save_every_n_iteration == 0:
                    time_step /= FLAGS.save_every_n_iteration
                    train_loss /= FLAGS.save_every_n_iteration
                    summary = tf.Summary()
                    summary.value.add(tag='train loss', simple_value=train_loss)
                    summary.value.add(tag='train ppl', simple_value=np.exp(train_loss))
                    train_writer.add_summary(summary=summary, global_step=global_step)
                    print("global step %d step-time %.4f train loss %f perplexity %f"
                          % (global_step, time_step, train_loss, np.exp(train_loss)))
                    train_loss = 0
                    time_step = 0

                    s2s.saver.save(sess, "%s/model.ckpt" % FLAGS.train_dir, global_step=global_step)

                    valid_loss = 0
                    for batch in dp.eval_batches(valid_data,FLAGS.batch_size):
                        [loss] = s2s.step(sess, batch, is_train=False)
                        valid_loss += loss
                    valid_loss /= FLAGS.valid_size // FLAGS.batch_size
                    summary = tf.Summary()
                    summary.value.add(tag='valid loss', simple_value=valid_loss)
                    summary.value.add(tag='valid ppl', simple_value=np.exp(valid_loss))
                    valid_writer.add_summary(summary, global_step)
                    print "valid loss:", valid_loss, "valid ppl:", np.exp(valid_loss)

        else:
            s2s.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
            dp.infer(s2s, sess)


if __name__ == '__main__':
    tf.app.run()
