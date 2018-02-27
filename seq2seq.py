import tensorflow as tf
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.python.layers.core import Dense
from tensorflow.python.framework import constant_op

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
    def __init__(self,
                 tfFLAGS,
                 embed=None):
        self.vocab_size = tfFLAGS.vocab_size
        self.embed_size = tfFLAGS.embed_size
        self.num_units = tfFLAGS.num_units
        self.num_layers = tfFLAGS.num_layers
        self.beam_width = tfFLAGS.beam_width
        self.use_lstm = tfFLAGS.use_lstm
        self.attn_mode = tfFLAGS.attn_mode
        self.share_emb = tfFLAGS.share_emb
        self.train_keep_prob = tfFLAGS.keep_prob
        self.max_decode_len = tfFLAGS.max_decode_len
        self.bi_encode = tfFLAGS.bi_encode
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.max_gradient_norm = 5.0
        with tf.variable_scope("optimizer"):
            if tfFLAGS.opt == 'SGD':
                self.learning_rate = tf.Variable(float(tfFLAGS.learning_rate),
                                                 trainable=False, dtype=tf.float32)
                self.learning_rate_decay_op = self.learning_rate.assign(
                    self.learning_rate * tfFLAGS.learning_rate_decay_factor)
                self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif tfFLAGS.opt == 'Momentum':
                self.opt = tf.train.MomentumOptimizer(learning_rate=tfFLAGS.learning_rate, momentum=tfFLAGS.momentum)
            else:
                self.opt = tf.train.AdamOptimizer()

        self._make_input(embed)

        with tf.variable_scope("input"):
            self.output_layer = Dense(self.vocab_size,
                                      # kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      use_bias=False)
        self._build_encoder()
        self._build_decoder()
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=1, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        for var in tf.trainable_variables():
            print var

    def _make_input(self, embed):
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
            default_value=_UNK,
            shared_name="out_table",
            name="out_table",
            checkpoint=True)
        with tf.variable_scope("input"):
            self.post_string = tf.placeholder(tf.string,(None,None),'post_string')
            self.response_string = tf.placeholder(tf.string, (None, None), 'response_string')

            self.post = self.symbol2index.lookup(self.post_string)
            self.post_len = tf.placeholder(tf.int32, (None,), 'post_len')
            self.response = self.symbol2index.lookup(self.response_string)
            self.response_len = tf.placeholder(tf.int32, (None,), 'response_len')

            with tf.variable_scope("embedding") as scope:
                if self.share_emb:
                    if embed is None:
                        # initialize the embedding randomly
                        self.emb_enc = self.emb_dec = tf.get_variable(
                            "emb_share", [self.vocab_size, self.embed_size], dtype=tf.float32
                        )
                    else:
                        # initialize the embedding by pre-trained word vectors
                        print "share pre-trained embed"
                        self.emb_enc = self.emb_dec = tf.get_variable('emb_share', dtype=tf.float32, initializer=embed)

                else:
                    if embed is None:
                        # initialize the embedding randomly
                        self.emb_enc = tf.get_variable(
                            "emb_enc", [self.vocab_size, self.embed_size], dtype=tf.float32
                        )
                        self.emb_dec = tf.get_variable(
                            "emb_dec", [self.vocab_size, self.embed_size], dtype=tf.float32
                        )
                    else:
                        # TODO initialize the embedding by pre-trained word vectors
                        self.emb_enc = tf.get_variable("emb_enc", dtype=tf.float32, initializer=embed)
                        self.emb_dec = tf.get_variable("emb_dec", dtype=tf.float32, initializer=embed)

            self.enc_inp = tf.nn.embedding_lookup(self.emb_enc, self.post)

            self.batch_len = tf.shape(self.response)[1]
            self.batch_size = tf.shape(self.response)[0]
            self.response_input = tf.concat([tf.ones((self.batch_size, 1), dtype=tf.int64) * GO_ID,
                                             tf.split(self.response, [self.batch_len - 1, 1], axis=1)[0]], 1)
            self.dec_inp = tf.nn.embedding_lookup(self.emb_dec, self.response_input)

            self.keep_prob = tf.placeholder_with_default(1.0, ())

    def _build_encoder(self):
        with tf.variable_scope("encode", initializer=tf.orthogonal_initializer()):
            if self.bi_encode:
                cell_fw, cell_bw = self._build_biencoder_cell()
                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=self.enc_inp,
                    sequence_length=self.post_len,
                    dtype=tf.float32
                )
                enc_outputs = tf.concat(outputs, axis=-1)
                enc_state = []
                for i in range(self.num_layers):
                    if self.use_lstm:
                        encoder_state_c = tf.concat([states[0][i].c,states[1][i].c], axis=-1)
                        encoder_state_h = tf.concat([states[0][i].h,states[1][i].h], axis=-1)
                        enc_state.append(tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h))
                    else:
                        enc_state.append(tf.concat([states[0][i],states[1][i]], axis=-1))
                enc_state = tuple(enc_state)
                self.enc_outputs, self.enc_state = enc_outputs, enc_state
            else:
                enc_cell = self._build_encoder_cell()
                enc_outputs, enc_state = tf.nn.dynamic_rnn(
                    cell=enc_cell,
                    inputs=self.enc_inp,
                    sequence_length=self.post_len,
                    dtype=tf.float32
                )
                self.enc_outputs, self.enc_state = enc_outputs, enc_state

    def _build_decoder(self):
        with tf.variable_scope("decode", initializer=tf.orthogonal_initializer()):
            dec_cell, init_state = self._build_decoder_cell(self.enc_outputs, self.post_len, self.enc_state)

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
                # maximum_iterations=self.max_decode_len,
            )
            logits = train_output.rnn_output

            mask = tf.sequence_mask(self.response_len, self.batch_len, dtype=tf.float32)

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.response, logits=logits)
            crossent = tf.reduce_sum(crossent * mask)
            # self.sen_loss = crossent / tf.to_float(self.batch_size)

            # ppl(loss avg) across each timestep, the same as :
            self.loss = tf.contrib.seq2seq.sequence_loss(train_output.rnn_output,
                                                         self.response,
                                                         mask)
            # self.loss = crossent / tf.reduce_sum(mask)
            self.sen_loss = self.loss

            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.sen_loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_op = self.opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

            self.train_out = self.index2symbol.lookup(tf.cast(train_output.sample_id, tf.int64), name='train_out')

        with tf.variable_scope("decode", reuse=True):
            dec_cell, init_state = self._build_decoder_cell(self.enc_outputs, self.post_len, self.enc_state,
                                                            alignment=True)

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
            infer_output, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=infer_decoder,
                maximum_iterations=self.max_decode_len,
            )

            self.alignment = tf.identity(final_context_state.alignment_history.stack(),
                                         name='alignment')

            self.inference = self.index2symbol.lookup(tf.cast(infer_output.sample_id, tf.int64), name='inference')

        with tf.variable_scope("decode", reuse=True):
            dec_init_state = tf.contrib.seq2seq.tile_batch(self.enc_state, self.beam_width)
            enc_outputs = tf.contrib.seq2seq.tile_batch(self.enc_outputs, self.beam_width)
            post_len = tf.contrib.seq2seq.tile_batch(self.post_len, self.beam_width)

            dec_cell, init_state = self._build_decoder_cell(enc_outputs, post_len, dec_init_state,
                                                            beam_width=self.beam_width)

            beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=dec_cell,
                embedding=self.emb_dec,
                start_tokens=tf.ones_like(self.post_len) * GO_ID,
                end_token=EOS_ID,
                initial_state=init_state,
                beam_width=self.beam_width,
                output_layer=self.output_layer
            )
            beam_output, _, beam_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=beam_decoder,
                maximum_iterations=self.max_decode_len,
            )

            self.beam_out = self.index2symbol.lookup(tf.cast(beam_output.predicted_ids, tf.int64), name='beam_out')

    def _build_encoder_cell(self):
        if self.use_lstm:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_units), self.keep_prob) for _ in range(self.num_layers)])
        else:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.num_units), self.keep_prob) for _ in range(self.num_layers)])
        return cell

    def _build_biencoder_cell(self):
        if self.use_lstm:
            cell_fw = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_units / 2), self.keep_prob) for _ in range(self.num_layers)])
            cell_bw = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_units / 2), self.keep_prob) for _ in range(self.num_layers)])
        else:
            cell_fw = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.num_units / 2), self.keep_prob) for _ in range(self.num_layers)])
            cell_bw = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.num_units / 2), self.keep_prob) for _ in range(self.num_layers)])
        return cell_fw, cell_bw

    def _build_decoder_cell(self, memory, memory_len, encode_state, beam_width=1, alignment=False):
        if self.use_lstm:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_units), self.keep_prob) for _ in range(self.num_layers)])
        else:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.num_units), self.keep_prob) for _ in range(self.num_layers)])
        if self.attn_mode=='Luong':
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.num_units,
                memory=memory,
                memory_sequence_length=memory_len,
                scale=True
            )
        elif self.attn_mode=='Bahdanau':
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.num_units,
                memory=memory,
                memory_sequence_length=memory_len,
                scale=True
            )
        else:
            return cell, encode_state
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.num_units,
            alignment_history=alignment,
        )
        return attn_cell, attn_cell.zero_state(self.batch_size * beam_width, tf.float32).clone(
            cell_state=encode_state)

    def initialize(self, sess, vocab):
        op_in = self.symbol2index.insert(constant_op.constant(vocab),
                                         constant_op.constant(range(len(vocab)), dtype=tf.int64))
        op_out = self.index2symbol.insert(constant_op.constant(range(len(vocab)), dtype=tf.int64),
                                          constant_op.constant(vocab))
        sess.run(tf.global_variables_initializer())
        sess.run([op_in, op_out])

    def step(self, sess, data, is_train=False):
        input_feed = {
            self.post_string: data['post'],
            self.post_len: data['post_len'],
            self.response_string: data['response'],
            self.response_len: data['response_len']
        }
        if is_train:
            output_feed = [self.train_op,
                           self.loss,
                           # self.post_string,
                           # self.response_string,
                           # self.train_out,
                           # self.inference,
                           # self.beam_out,
                           ]
            input_feed[self.keep_prob] = self.train_keep_prob
        else:
            output_feed = [self.loss,
                           # self.post_string,
                           # self.response_string,
                           # self.train_out,
                           # self.inference,
                           # self.beam_out,
                           ]
        return sess.run(output_feed, input_feed)
