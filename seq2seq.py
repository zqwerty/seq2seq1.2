import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.python.layers.core import Dense
from tensorflow.python.framework import constant_op


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_PAD = b"PAD"
_GO = b"GO"
_EOS = b"EOS"
_UNK = b"UNK"
_START_VOCAB = [_PAD, _EOS, _GO, _UNK]

PAD_ID = 0
EOS_ID = 1
GO_ID = 2
UNK_ID = 3


class seq2seq(object):
    def __init__(self, hparams_dict):
        # TODO: copy hparams
        # self.batch_size = hparams_dict['batch_size']
        self.vocab_size = hparams_dict['vocab_size']
        self.embed_size = hparams_dict['embed_size']
        self.num_units = hparams_dict['num_units']
        self.num_layers = hparams_dict['num_layers']
        self.beam_width = hparams_dict['beam_width']
        self.use_lstm = hparams_dict['use_lstm']
        self.attn_mode = hparams_dict['attn_mode']
        self.share_emb = hparams_dict['share_emb']
        self.lr = hparams_dict['lr']
        self.keep_prob = hparams_dict['keep_prob']
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self._make_input()

        self.output_layer = Dense(self.vocab_size,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        self._build_encoder()
        self._build_decoder()
        self.saver = tf.train.Saver()
        print hparams_dict


    def _make_input(self):
        self.symbol2index = MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=UNK_ID,
            shared_name="in_table",
            name="in_table",
            checkpoint=True)
        self.index2symbol = MutableHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.string,
            default_value='_UNK',
            shared_name="out_table",
            name="out_table",
            checkpoint=True)

        self.post_string = tf.placeholder(tf.string,(None,None),'post')
        self.response_string = tf.placeholder(tf.string, (None, None), 'response')

        self.post = self.symbol2index.lookup(self.post_string)
        self.post_len = tf.placeholder(tf.int32, (None,), 'post_len')
        self.response = self.symbol2index.lookup(self.response_string)
        self.response_len = tf.placeholder(tf.int32, (None,), 'response_len')

        with tf.variable_scope("embedding") as scope:
            if self.share_emb:
                self.emb_enc = self.emb_dec = tf.get_variable(
                    "emb_share", [self.vocab_size, self.embed_size], dtype=tf.float32
                )
            else:
                self.emb_enc = tf.get_variable(
                    "emb_enc", [self.vocab_size, self.embed_size], dtype=tf.float32
                )
                self.emb_dec = tf.get_variable(
                    "emb_dec", [self.vocab_size, self.embed_size], dtype=tf.float32
                )

        self.enc_inp = tf.nn.embedding_lookup(self.emb_enc, self.post)

        self.batch_len = tf.shape(self.response)[1]
        self.batch_size = tf.shape(self.response)[0]
        self.response_input = tf.concat([tf.ones((self.batch_size, 1), dtype=tf.int64) * GO_ID,
                                         tf.split(self.response, [self.batch_len - 1, 1], axis=1)[0]], 1)
        self.dec_inp = tf.nn.embedding_lookup(self.emb_dec, self.response_input)


    def _build_encoder(self):
        with tf.variable_scope("encode"):
            enc_cell = self._build_encoder_cell()
            self.enc_outputs, self.enc_state = tf.nn.dynamic_rnn(
                cell=enc_cell,
                inputs=self.enc_inp,
                sequence_length=self.post_len,
                dtype=tf.float32
            )

    def _build_decoder(self):
        dec_init_state = self.enc_state

        with tf.variable_scope("decode"):
            dec_cell = self._build_decoder_cell(self.enc_outputs, self.post_len)
            init_state = dec_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=dec_init_state)
            train_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.dec_inp,
                sequence_length=self.response_len
            )
            train_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell,
                helper=train_helper,
                initial_state=init_state,
                output_layer=self.output_layer
            )
            train_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=train_decoder,
            )

        mask = tf.sequence_mask(self.response_len, self.batch_len, dtype=tf.float32)
        self.train_loss = tf.contrib.seq2seq.sequence_loss(train_output.rnn_output, self.response, mask)
        # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.train_loss)
        self.train_out = self.index2symbol.lookup(tf.cast(train_output.sample_id,tf.int64))

        # calculate the gradient of parameters
        self.params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(1)
        gradients = tf.gradients(self.train_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
                                                                       5.0)
        self.train_op = opt.apply_gradients(zip(clipped_gradients, self.params))

        with tf.variable_scope("decode", reuse=True):
            dec_cell = self._build_decoder_cell(self.enc_outputs, self.post_len)
            init_state = dec_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=dec_init_state)

            start_tokens = tf.tile(tf.constant([GO_ID], dtype=tf.int32), [self.batch_size])
            end_token = EOS_ID
            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.emb_dec,
                start_tokens,
                end_token
            )
            infer_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell,
                helper=infer_helper,
                initial_state=init_state,
                output_layer=self.output_layer
            )
            infer_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=infer_decoder
            )

        self.inference = self.index2symbol.lookup(tf.cast(infer_output.sample_id,tf.int64))

        dec_init_state = tf.contrib.seq2seq.tile_batch(dec_init_state, self.beam_width)
        enc_outputs = tf.contrib.seq2seq.tile_batch(self.enc_outputs, self.beam_width)
        post_len = tf.contrib.seq2seq.tile_batch(self.post_len, self.beam_width)

        with tf.variable_scope("decode", reuse=True):
            dec_cell = self._build_decoder_cell(enc_outputs, post_len)
            init_state = dec_cell.zero_state(self.batch_size * self.beam_width, tf.float32).clone(
                cell_state=dec_init_state)

            test_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=dec_cell,
                embedding=self.emb_dec,
                start_tokens=tf.ones_like(self.post_len) * GO_ID,
                end_token=EOS_ID,
                initial_state=init_state,
                beam_width=self.beam_width,
                output_layer=self.output_layer
            )
            test_output, _, test_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=test_decoder,
            )

        self.test_len = tf.shape(test_output.predicted_ids)
        self.test_out = self.index2symbol.lookup(tf.cast(test_output.predicted_ids,tf.int64))

    def _build_encoder_cell(self):
        if self.use_lstm:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_units),self.keep_prob) for _ in range(self.num_layers)])
        else:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.num_units),self.keep_prob) for _ in range(self.num_layers)])
        return cell

    def _build_decoder_cell(self,memory,memory_len):
        if self.use_lstm:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_units),self.keep_prob) for _ in range(self.num_layers)])
        else:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.num_units),self.keep_prob) for _ in range(self.num_layers)])
        if (self.attn_mode=='Luong'):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.num_units,
                memory=memory,
                memory_sequence_length=memory_len,
                scale=True
            )
        elif (self.attn_mode=='Bahdanau'):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.num_units,
                memory=memory,
                memory_sequence_length=memory_len,
                scale=True
            )
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.num_units,
        )
        return attn_cell

    def infer(self, sess, dp):
        while True:
            infer_data = {}
            infer_data['post'] = raw_input('post > ').strip().split()
            infer_data['response'] = ''.strip().split()
            infer_data = [infer_data]
            for batch in dp.batchify(infer_data,1):
                res = sess.run([self.inference,self.test_out],batch)
                print 'inference > '+' '.join(res[0][0])
                print 'beam > '+' '.join(res[1][0,:,0])

