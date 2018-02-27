import random
from itertools import izip
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op
import time
from seq2seq import seq2seq
from utils import load_data, train_batch, eval_batches, build_vocab, infer
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_dir", "/home/data/zhuqi/weibo_pair/", "data_dir")
tf.app.flags.DEFINE_string("data_train", "train2.weibo_pair", "train data")
tf.app.flags.DEFINE_string("data_valid", "valid.weibo_pair", "valid data")
tf.app.flags.DEFINE_string("data_test", "test.weibo_pair", "test data")
tf.app.flags.DEFINE_string("infer_out", "infer_out.txt", "infer out")

tf.app.flags.DEFINE_string("word_vector", "/home/data/zhuqi/vector.txt", "word vector")

tf.app.flags.DEFINE_string("train_dir", "/home/data/zhuqi/model_log/seq2seq1.2/train/train173", "train_dir")
tf.app.flags.DEFINE_string("log_dir", "/home/data/zhuqi/model_log/seq2seq1.2/log/log173", "log_dir")
tf.app.flags.DEFINE_string("save_para_path", "", "path of the trained model, default latest in train_dir")

tf.app.flags.DEFINE_string("attn_mode", "Luong", "attn_mode")
tf.app.flags.DEFINE_string("opt", "SGD", "optimizer")

tf.app.flags.DEFINE_boolean("is_train", True, "is_train")
tf.app.flags.DEFINE_boolean("use_lstm", False, "lstm/GRU")
tf.app.flags.DEFINE_boolean("bi_encode", False, "bidirectional encoder")
tf.app.flags.DEFINE_boolean("share_emb", True, "share_emb")

tf.app.flags.DEFINE_integer("batch_size", 128, "batch_size")
tf.app.flags.DEFINE_integer("embed_size", 100, "embed_size")
tf.app.flags.DEFINE_integer("num_units", 512, "num_units")
tf.app.flags.DEFINE_integer("num_layers", 2, "num_layers")

tf.app.flags.DEFINE_integer("beam_width", 5, "beam_width")
tf.app.flags.DEFINE_integer("max_decode_len", 128, "max_decode_len")
tf.app.flags.DEFINE_integer("vocab_size", 40000, "vocab_size")
tf.app.flags.DEFINE_integer("save_every_n_iteration", 1000, "save_every_n_iteration")
tf.app.flags.DEFINE_integer("max_step", 300000, "max_step")

tf.app.flags.DEFINE_float("learning_rate", 0.5, "learning rate")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "learning rate decay factor")
tf.app.flags.DEFINE_float("momentum", 0.9, "momentum")
tf.app.flags.DEFINE_float("keep_prob", 0.8, "keep_prob")


def main(unused_argv):
    print(FLAGS.__flags)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if FLAGS.is_train:
            train_data = load_data(os.path.join(FLAGS.data_dir, FLAGS.data_train) + '.post',
                                   os.path.join(FLAGS.data_dir, FLAGS.data_train) + '.response')
            valid_data = load_data(os.path.join(FLAGS.data_dir, FLAGS.data_valid) + '.post',
                                   os.path.join(FLAGS.data_dir, FLAGS.data_valid) + '.response')

            if tf.train.get_checkpoint_state(FLAGS.train_dir):
                print("Reading model parameters from %s" % FLAGS.train_dir)
                model = seq2seq(tfFLAGS=FLAGS)
                model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
                train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'))
            else:
                print("Create new parameters")
                vocab, embed = build_vocab(FLAGS.word_vector, FLAGS.embed_size, FLAGS.vocab_size, train_data)
                model = seq2seq(tfFLAGS=FLAGS, embed=embed)
                model.initialize(sess, vocab=vocab)
                train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'), sess.graph)

            valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'valid'))
            loss_placeholder = tf.placeholder(tf.float32)
            loss_summary_op = tf.summary.scalar('loss', loss_placeholder)
            ppl_op = tf.exp(loss_placeholder)
            ppl_summary_op = tf.summary.scalar('ppl', ppl_op)

            train_loss = 0
            time_step = 0
            previous_losses = [1e18] * 3
            while True:
                start_time = time.time()

                train_batched = train_batch(train_data, FLAGS.batch_size)
                train_op, loss = model.step(sess, train_batched, is_train=True)

                global_step = model.global_step.eval()
                train_loss += loss
                time_step += (time.time() - start_time)

                if global_step % FLAGS.save_every_n_iteration == 0:
                    time_step /= FLAGS.save_every_n_iteration
                    train_loss /= FLAGS.save_every_n_iteration

                    if FLAGS.opt == 'SGD' and train_loss > max(previous_losses):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses = previous_losses[1:] + [train_loss]

                    loss, ppl = sess.run([loss_summary_op, ppl_summary_op],
                                         feed_dict={loss_placeholder:train_loss})
                    train_writer.add_summary(summary=loss, global_step=global_step)
                    train_writer.add_summary(summary=ppl, global_step=global_step)
                    summary_lr = tf.Summary(value=[tf.Summary.Value(tag="lr", simple_value=model.learning_rate.eval())])
                    train_writer.add_summary(summary=summary_lr, global_step=global_step)
                    print("global step %d step-time %.4f train loss %f perplexity %f learning_rate %f"
                          % (global_step,
                             time_step,
                             train_loss,
                             np.exp(train_loss),
                             model.learning_rate.eval() if FLAGS.opt=='SGD' else .0))
                    train_loss = 0
                    time_step = 0

                    valid_loss = 0
                    for batch in eval_batches(valid_data,FLAGS.batch_size):
                        [loss] = model.step(sess, batch, is_train=False)
                        valid_loss += loss
                    valid_loss /= len(valid_data) // FLAGS.batch_size

                    loss, ppl = sess.run([loss_summary_op, ppl_summary_op],
                                         feed_dict={loss_placeholder: valid_loss})
                    valid_writer.add_summary(summary=loss, global_step=global_step)
                    valid_writer.add_summary(summary=ppl, global_step=global_step)
                    print "valid loss:", valid_loss, "valid ppl:", np.exp(valid_loss)

                    model.saver.save(sess, "%s/model.ckpt" % FLAGS.train_dir, global_step=global_step)

                if global_step >= FLAGS.max_step:
                    break
        else:
            if FLAGS.save_para_path=='':
                model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            else:
                model_path = FLAGS.save_para_path
            print model_path
            saver = tf.train.import_meta_graph(model_path + '.meta')
            saver.restore(sess, model_path)
            test_data = load_data(os.path.join(FLAGS.data_dir, FLAGS.data_test) + '.post',
                                  os.path.join(FLAGS.data_dir, FLAGS.data_test) + '.response')
            infer(sess, test_data, FLAGS.batch_size, os.path.join(FLAGS.data_dir, FLAGS.infer_out))


if __name__ == '__main__':
    tf.app.run()
