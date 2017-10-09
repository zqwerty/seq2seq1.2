import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.python.layers.core import Dense
from tensorflow.python.framework import constant_op


tokens = {"PAD": 0, "EOS": 1, "GO": 2, "UNK": 3}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_PAD = b"PAD"
_GO = b"GO"
_EOS = b"EOS"
_UNK = b"UNK"
_START_VOCAB = [_PAD, _EOS, _GO, _UNK]

PAD_ID = 0
EOS_ID = 1
GO_ID = 2
UNK_ID = 3

class seq2seqModel(object):
    def __init__(self,
                 num_layers,
                 num_units,
                 num_symbols,
                 num_embed_units,
                 use_lstm=False,
                 attention_option="Luong",
                 encoder_type="uni",
                 embed=None,
                 embed_share=False,
                 mode="train"
                 ):
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_symbols = num_symbols
        self.num_embed_unit = num_embed_units
        self.use_lstm = use_lstm
        self.attention_option = attention_option
        self.encoder_type = encoder_type
        self.mode = mode

        with tf.variable_scope("embedding") as scope:
            if embed_share:
                if embed is None:
                    # initialize the embedding randomly
                    embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
                else:
                    # initialize the embedding by pre-trained word vectors
                    embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
                self.embedding_encoder = embed
                self.embedding_decoder = embed
            else:
                with tf.variable_scope("encoder"):
                    self.embedding_encoder = tf.get_variable(
                        "embedding_encoder", [num_symbols, num_embed_units], dtype=tf.float32)

                with tf.variable_scope("decoder"):
                    self.embedding_decoder = tf.get_variable(
                        "embedding_decoder", [num_symbols, num_embed_units], dtype=tf.float32)

        self.output_layer = Dense(self.num_symbols,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))



    def _build_encoder_cell(self):
        if self.use_lstm:
            return tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.LSTMCell(self.num_units) for _ in range(self.num_layers)]
            )
        else:
            return tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(self.num_units) for _ in range(self.num_layers)]
            )

    def _build_decoder_cell(self,
                            encoder_outputs,
                            source_sequence_length,
                            ):
        """Create attention mechanism based on the attention_option."""
        # Mechanism
        if self.attention_option == "luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.num_units,
                encoder_outputs,
                memory_sequence_length=source_sequence_length)
        elif self.attention_option == "scaled_luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.num_units,
                encoder_outputs,
                memory_sequence_length=source_sequence_length,
                scale=True)
        elif self.attention_option == "bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.num_units,
                encoder_outputs,
                memory_sequence_length=source_sequence_length)
        elif self.attention_option == "normed_bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.num_units,
                encoder_outputs,
                memory_sequence_length=source_sequence_length,
                normalize=True)
        else:
            # raise ValueError("Unknown attention option %s" % attention_option)
            attention_mechanism = None

        if self.use_lstm:
            cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.LSTMCell(self.num_units) for _ in range(self.num_layers)]
            )
        else:
            cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(self.num_units) for _ in range(self.num_layers)]
            )

        if attention_mechanism is not None:
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell,
                attention_mechanism,
                attention_layer_size=self.num_units,
            )
        else:
            decoder_cell = cell
        return decoder_cell

    def _build_encoder(self,
                       encoder_inputs,
                       source_sequence_length,
                       ):
        with tf.variable_scope("encoder") as scope:
            encoder_emb_inputs = tf.nn.embedding_lookup(self.embedding_encoder,
                                                        encoder_inputs)

            if self.encoder_type == "uni":
                cell = self._build_encoder_cell()
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,
                    encoder_emb_inputs,
                    dtype=tf.float32,
                    sequence_length=source_sequence_length,
                )
            elif self.encoder_type == "bi":
                # TODO
                pass
            else:
                # TODO
                pass
        return encoder_outputs, encoder_state


    def _build_decoder(self,
                       encoder_state,
                       encoder_outputs,
                       source_sequence_length,
                       decoder_inputs,
                       target_sequence_length,
                       ):

        with tf.variable_scope("decode") as decoder_scope:
            # maybe some bug
            decoder_init_state = encoder_state

            decoder_cell = self._build_decoder_cell(encoder_outputs, source_sequence_length)


            if self.mode == "train":
                decoder_emb_inputs = tf.nn.embedding_lookup(self.embedding_decoder,
                                                            decoder_inputs)
                train_helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inputs,
                    target_sequence_length,
                )

                # decoder
                train_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    train_helper,
                    decoder_init_state,
                )

                #  dynamic decoding
                self.decoder_output, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    train_decoder,
                    scope=decoder_scope
                )

                sample_id = self.decoder_output.sample_id
                logits = self.output_layer(self.decoder_output.rnn_output)
            elif self.mode == "inference":
                # TODO
                pass
        return logits, sample_id, final_context_state

    def make_inputs(self):
        self.posts = tf.placeholder(tf.string, [None, None], 'enc_inps')  # batch*len
        self.posts_length = tf.placeholder(tf.int32, [None], 'enc_lens')  # batch
        self.responses = tf.placeholder(tf.string, [None, None], 'dec_inps')  # batch*len
        self.responses_length = tf.placeholder(tf.int32, [None], 'dec_lens')  # batch

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

        self.posts_input = self.symbol2index.lookup(self.posts)  # batch*len
        self.responses_target = self.symbol2index.lookup(self.responses)  # batch*len
        self.batch_size, self.decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        self.responses_input = tf.concat([tf.ones([self.batch_size, 1], dtype=tf.int64) * GO_ID,
                                          tf.split(self.responses_target, [decoder_len - 1, 1], 1)[0]], 1)  # batch*len


    def _build_graph(self):
        with tf.variable_scope("dynamic_seq2seq", dtype=tf.float32):
            encoder_outputs, encoder_state = self._build_encoder(self.posts_input, self.posts_length)

            logits, sample_id, final_context_state = self._build_decoder(
                encoder_state,
                encoder_outputs,
                self.posts_length,
                self.responses_input,
                self.responses_length
            )

            if self.mode == "train":
                loss = self._compute_loss(logits)

        return logits, loss, final_context_state, sample_id


    def _compute_loss(self, logits):
        max_time = self.decoder_len
        masks = tf.sequence_mask(self.responses_length, max_time, dtype=tf.float32)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.responses_target,
            logits=logits
        )
        return tf.reduce_sum(crossent * masks) / tf.to_float(self.batch_size)
        # self.decoder_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=self.responses_target,
        #                                                      weights=masks)

    def train_step(self):
        pass

    def inference(self):
        pass


class seq2seq(object):
    def __init__(self, hparams_dict):
        # TODO: copy hparams
        self.batch_size = hparams_dict['batch_size']
        self.vocab_size = hparams_dict['vocab_size']
        self.embed_size = hparams_dict['embed_size']
        self.num_units = hparams_dict['num_units']
        self.num_layers = hparams_dict['num_layers']
        self.beam_width = hparams_dict['beam_width']
        self.use_lstm = hparams_dict['use_lstm']
        self.attn_mode = hparams_dict['attn_mode']
        self.share_emb = hparams_dict['share_emb']
        self.lr = hparams_dict['lr']


        self.make_input()

        self.output_layer = Dense(self.vocab_size,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        self._build_encoder()
        self._build_decoder()
        print hparams_dict
        self.params = tf.trainable_variables()

    def make_input(self):
        self.post_id = tf.placeholder(tf.int64, (None, None), 'post')
        self.response_id = tf.placeholder(tf.int64, (None, None), 'response')



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

        self.post_string = self.index2symbol.lookup(self.post_id)
        self.response_string = self.index2symbol.lookup(self.response_id)

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
        self.response_input = tf.concat([tf.ones((self.batch_size, 1), dtype=tf.int64) * tokens["GO"],
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
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.train_loss)
        self.train_out = self.index2symbol.lookup(tf.cast(train_output.sample_id,tf.int64))

        with tf.variable_scope("decode", reuse=True):
            dec_cell = self._build_decoder_cell(self.enc_outputs, self.post_len)
            init_state = dec_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=dec_init_state)

            start_tokens = tf.tile(tf.constant([tokens['GO']], dtype=tf.int32), [self.batch_size])
            end_token = tokens["EOS"]
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
                start_tokens=tf.ones_like(self.post_len) * tokens["GO"],
                end_token=tokens["EOS"],
                initial_state=init_state,
                beam_width=self.beam_width,
                output_layer=self.output_layer
            )
            test_output, _, test_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=test_decoder,
                maximum_iterations=10)

        self.test_len = tf.shape(test_output.predicted_ids)
        self.test_out = self.index2symbol.lookup(tf.cast(test_output.predicted_ids,tf.int64))
        predicts = self.to_sparse(test_output.predicted_ids[:, :, 0], test_lengths[:, 0])
        labels = self.to_sparse(self.response, self.response_len)
        self.error_rate = tf.reduce_mean(tf.edit_distance(predicts, labels))

    def _build_encoder_cell(self):
        if self.use_lstm:
            return tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.num_units) for _ in range(self.num_layers)])
        else:
            return tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.num_units) for _ in range(self.num_layers)])

    def _build_decoder_cell(self,memory,memory_len):
        if self.use_lstm:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.num_units) for _ in range(self.num_layers)])
        else:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.num_units) for _ in range(self.num_layers)])
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

    # Convert a dense matrix into a sparse matrix (for e.g. edit_distance)
    def to_sparse(self, tensor, lengths):
        mask = tf.sequence_mask(lengths, 10)
        indices = tf.to_int64(tf.where(tf.equal(mask, True)))
        values = tf.to_int64(tf.boolean_mask(tensor, mask))
        shape = tf.to_int64(tf.shape(tensor))
        return tf.SparseTensor(indices, values, shape)

# Divide training samples into batches
def batchify(s2s):

    for i in range(10000 // s2s.batch_size):
        yield next_batch(s2s,i)

# Create a single batch at i * batch_size
def next_batch(s2s, i):

    start = i * s2s.batch_size
    stop = (i+1) * s2s.batch_size

    batch = {
            'post:0':     post[start:stop],
            s2s.post_len:   post_len[start:stop],
            'response:0':    response[start:stop],
            s2s.response_len: response_len[start:stop]
    }

    return batch

def test_batch(s2s):
    tp, tpl, tr, trl = make_inputs()
    return {
        'post:0': tp[:50],
        s2s.post_len: tpl[:50],
        'response:0': tr[:50],
        s2s.response_len: trl[:50]
    }






def make_inputs():

    minLength = 5
    maxLength = 10
    samples = 10000
    vocab_size = 50

    post = np.random.randint(
        low=len(tokens),
        high=vocab_size,
        size=(samples, maxLength)
    )
    post_len = np.random.randint(
        low=minLength,
        high=maxLength,
        size=samples,
    )
    response = np.ones_like(post)*tokens["PAD"]
    for sample_idx in xrange(post.shape[0]):
        post[sample_idx, post_len[sample_idx]:] = tokens["PAD"]
        response[sample_idx,:post_len[sample_idx]] = np.sort(post[sample_idx,:post_len[sample_idx]])
        response[sample_idx, post_len[sample_idx]] = tokens["EOS"]

    response_len = np.copy(post_len+1)
    return post, post_len, response, response_len

post, post_len, response, response_len = make_inputs()
print post.shape
h_dict = {}
h_dict['batch_size'] = 50
h_dict['vocab_size'] = 50
h_dict['embed_size'] = 15
h_dict['num_units'] = 100
h_dict['num_layers'] = 2
h_dict['beam_width'] = 4
h_dict['use_lstm'] = True
h_dict['attn_mode'] = 'Luong'
h_dict['share_emb'] = True
h_dict['data_dir'] = 'data'
h_dict['splited'] = False
h_dict['lr'] = 0.001
s2s = seq2seq(h_dict)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    vocab = map(str,range(50))
    op_in = s2s.symbol2index.insert(constant_op.constant(vocab),
                                constant_op.constant(range(h_dict['vocab_size']), dtype=tf.int64))
    op_out = s2s.index2symbol.insert(constant_op.constant(range(h_dict['vocab_size']), dtype=tf.int64),
                                     constant_op.constant(vocab))
    sess.run([op_in,op_out])
    for epoch in range(50):

        # Keep track of average train cost for this epoch
        train_cost = 0
        for batch in batchify(s2s):
            # print batch
            res =  sess.run([s2s.train_op, s2s.train_loss,
                                    ], batch)
            train_cost += res[1]
        # print res[9][0]
        train_cost /= 10000 / s2s.batch_size

        # Test time
        t = sess.run([s2s.error_rate,
                      s2s.post,
                      s2s.response,
                      s2s.train_out,
                      s2s.inference,
                      s2s.test_out,
                      s2s.train_loss], test_batch(s2s))
        er = t[0]
        print 'post: ',t[1][0]
        print 'response: ',t[2][0]
        print 'train_out: ',t[3][0]
        print 'infer_out: ',t[4][0]
        print 'test_out: ',t[5][0,:,0]
        print 'ppl: ',np.exp(t[6])

        print("Epoch", (epoch + 1), "train loss:", train_cost, "test error:", er)


