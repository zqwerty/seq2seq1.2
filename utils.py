import tensorflow as tf
import numpy as np
from itertools import izip
import random
import os

_PAD = b"PAD"
_GO = b"GO"
_EOS = b"EOS"
_UNK = b"UNK"
_START_VOCAB = [_PAD, _EOS, _GO, _UNK]

PAD_ID = 0
EOS_ID = 1
GO_ID = 2
UNK_ID = 3


def load_data(post_f, resp_f):
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


def build_vocab(word_vector_path, embed_size, vocab_size, data):
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
    if len(vocab_list) > vocab_size:
        vocab_list = vocab_list[:vocab_size]

    if not os.path.exists(word_vector_path):
        print("Cannot find word vectors")
        embed = None
        return vocab_list, embed

    print("Loading word vectors...")
    vectors = {}
    with open(word_vector_path) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ') + 1:]
            vectors[word] = vector

    embed = []
    pre_vector = 0
    for word in vocab_list:
        if word in vectors:
            vector = map(float, vectors[word].split())
            pre_vector += 1
        else:
            vector = np.zeros(embed_size, dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
    print("%d word vectors pre-trained" % pre_vector)

    return vocab_list, embed


def train_batch(data, batch_size):
    selected_data = [random.choice(data) for _ in range(batch_size)]
    batched_data = gen_batched_data(selected_data)
    return batched_data


def eval_batches(data, batch_size):
    for i in range(len(data)//batch_size):
        batched_data = gen_batched_data(data[i*batch_size:(i+1)*batch_size])
        yield batched_data


def gen_batched_data(data):
    post_max_len = max([len(item['post']) for item in data]) + 1
    response_max_len = max([len(item['response']) for item in data]) + 1

    posts, responses, posts_length, responses_length = [], [], [], []

    def padding(sent, l):
        return sent + [_EOS] + [_PAD] * (l - len(sent) - 1)

    for item in data:
        posts.append(padding(item['post'], post_max_len))
        responses.append(padding(item['response'], response_max_len))
        posts_length.append(len(item['post']) + 1)
        responses_length.append(len(item['response']) + 1)

    batched_data = {'post': np.array(posts),
                    'response': np.array(responses),
                    'post_len': posts_length,
                    'response_len': responses_length,
                    }
    return batched_data


def cut_eos(sentence):
    if sentence.find('EOS') != -1:
        return sentence[:sentence.find('EOS')]
    return sentence


def infer(sess, data, batch_size, infer_out):
    """
    TODO: add attention alignment
    """
    id = 0
    f = open(infer_out, 'wb')
    for j in range((len(data) + batch_size - 1) // batch_size):
        batch = gen_batched_data(data[j * batch_size:(j + 1) * batch_size])
        input_feed = {
            'input/post_string:0': batch['post'],
            'input/post_len:0': batch['post_len'],
            'input/response_string:0': batch['response'],
            'input/response_len:0': batch['response_len']
        }
        res = sess.run(['decode_1/inference:0', 'decode_2/beam_out:0'], input_feed)
        print id
        for i in range(len(batch['post_len'])):
            print >> f, 'post: ' + ' '.join(data[id]['post'])
            print >> f, 'response: ' + ' '.join(data[id]['response'])
            print >> f, 'infer: ' + cut_eos(' '.join(res[0][i]))
            print >> f, 'beam: ' + cut_eos(' '.join(res[1][i, :, 0]))
            print >> f, ''

            id += 1
    f.close()


if __name__ == '__main__':
    data = load_data('/home/data/zhuqi/weibo_pair/test.weibo_pair.post',
                     '/home/data/zhuqi/weibo_pair/test.weibo_pair.response')
    data = data[:10]
    for i in range(5):
        print 'post: ' + ' '.join(data[i]['post'])
        print 'response: ' + ' '.join(data[i]['response'])