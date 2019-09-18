from vocab import *
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import *
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
from tensorflow.contrib.rnn import LSTMStateTuple
import argparse
import time
from relational_memory import *
from data_utils import *
_model_path = os.path.join(param.get_data_dir(), 'model')
data_dir = param.get_data_dir()
save_dir = param.get_save_dir()
_NUM_UNITS = param.NUM_UNITS#输入宽度
if_segment = param.if_segment()

log_device_placement=True#是否打印设备分配日志
allow_soft_placement=True#如果你指定的设备不存在，允许TF自动分配设备
tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)


class Generator:
    def __init__(self,encoder_inputs,encoder_lengths,encoder_inputs_2,encoder_lengths_2,decoder_inputs,
                     decoder_lengths,_embed_ph,learn_rate,start_token,if_test = False,temperature = None,end_rate = 1):
        self.start_tokens = tf.constant([start_token] * BATCH_SIZE, dtype=tf.int32)

        self.learn_rate = learn_rate
        keep_prob = 0.8

        with tf.variable_scope('generator',initializer=tf.orthogonal_initializer()):
            with tf.variable_scope('embedding'):
                self.embedding = tf.get_variable(name='embedding', shape=[param.VOCAB_SIZE, INPUT_DIM], trainable=True)
                self._embed_ph = _embed_ph
                self._embed_init = self.embedding.assign(self._embed_ph)
            self.encoder_inputs = encoder_inputs
            self.encoder_lengths = encoder_lengths
            if param.Use_VAE:
                self.encoder_inputs_2 = encoder_inputs_2
                self.encoder_lengths_2 = encoder_lengths_2

            self.decoder_lengths = decoder_lengths
            self.decoder_inputs = decoder_inputs[:,:-1]
            max_len = tf.shape(decoder_inputs)[-1]

            with tf.variable_scope('cell', initializer=xavier_initializer()):
                self.encoder_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(_NUM_UNITS,activation=tf.tanh) for _ in range(param.NUM_LAYERS)])
                self.encoder_init_state = self.encoder_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

                _, self.encoder_final_state = tf.nn.dynamic_rnn(
                        cell=self.encoder_cell,
                        initial_state=self.encoder_init_state,
                        inputs=self.encoder_inputs,
                        sequence_length=self.encoder_lengths,
                        scope='encoder')
                if param.Use_VAE:
                    self.encoder_cell_2 = rnn.MultiRNNCell(
                        [rnn.BasicLSTMCell(_NUM_UNITS, activation=tf.tanh) for _ in range(param.NUM_LAYERS)])
                    self.encoder_init_state_2 = self.encoder_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
                    _, self.encoder_final_state_2 = tf.nn.dynamic_rnn(
                        cell=self.encoder_cell_2,
                        initial_state=self.encoder_init_state_2,
                        inputs= tf.nn.embedding_lookup(self.embedding, self.decoder_inputs),
                        sequence_length=self.decoder_lengths,
                        scope='encoder_2')




                self.decoder_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(_NUM_UNITS) for _ in range(param.NUM_LAYERS)])
                self.decoder_cell_drop = tf.contrib.rnn.DropoutWrapper(self.decoder_cell, output_keep_prob=keep_prob)
                self.decoder_init_state = self.decoder_cell_drop.zero_state(BATCH_SIZE,dtype=tf.float32)
                if param.Use_VAE:
                    self.decoder_input_state_2 = self.encoder_final_state_2
                    # self.decoder_input_state = (LSTMStateTuple(self.decoder_input_state[0,0],self.decoder_input_state[0,1]),
                    #                         LSTMStateTuple(self.decoder_input_state[1,0],self.decoder_input_state[1,1]))

                self.decoder_input_state = self.encoder_final_state

                if Use_relational_memory:
                    g_output_unit = create_output_unit(_NUM_UNITS * NUM_LAYERS * 2, param.VOCAB_SIZE)
                    mem_slots = param.mem_slots
                    head_size = param.head_size
                    num_heads = param.num_heads
                    gen_mem = RelationalMemory(mem_slots=mem_slots, head_size=head_size, num_heads=num_heads)
                    self.decoder_inputs = tf.pad(self.decoder_inputs,[[0,0],[0,1]])
                    x_emb = tf.transpose(tf.nn.embedding_lookup(self.embedding, self.decoder_inputs),
                                         perm=[1, 0, 2])  # seq_len x batch_size x emb_dim
                    g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=max_len, dynamic_size=True,
                                                                 infer_shape=True)
                    g_predictions = g_predictions.write(0,tf.one_hot(self.decoder_inputs[:,0],param.VOCAB_SIZE))
                    ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=max_len)
                    ta_emb_x = ta_emb_x.unstack(x_emb)
                    with tf.variable_scope('postprocessing', initializer=xavier_initializer()):
                        if Post_with_state:
                            self.softmax_w = tf.get_variable('softmax_w', [head_size*num_heads + 2 * NUM_UNITS, param.VOCAB_SIZE])
                        else:
                            self.softmax_w = tf.get_variable('softmax_w', [head_size*num_heads, param.VOCAB_SIZE])

                        self.softmax_b = tf.get_variable('softmax_b', [param.VOCAB_SIZE])
                    # the generator recurrent moddule used for pre-training
                    def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
                        mem_o_t, h_t = gen_mem(x_t, h_tm1)
                        if Post_with_state:
                            mem_o_t = tf.concat([tf.reshape(mem_o_t, [-1, head_size * num_heads]),
                                                 self.decoder_input_state[:,0,:2*NUM_UNITS]],-1)
                        else:
                            mem_o_t = tf.reshape(mem_o_t, [-1, head_size * num_heads])
                        o_t = tf.nn.bias_add(
                            tf.matmul(mem_o_t, self.softmax_w),
                            bias=self.softmax_b)
                        #o_t = g_output_unit(mem_o_t)
                        g_predictions = g_predictions.write(i, o_t)  # batch_size x vocab_size
                        x_tp1 = ta_emb_x.read(i)
                        return i + 1, x_tp1, h_t, g_predictions

                    self.decoder_input_state = tf.convert_to_tensor(self.decoder_input_state)
                    self.decoder_input_state = tf.transpose(self.decoder_input_state,
                                         perm=[2, 0, 1, 3])
                    self.decoder_input_state = tf.reshape(self.decoder_input_state,[BATCH_SIZE, mem_slots, -1])


                    if param.Use_latent_z:
                        self.decoder_input_state = tf.concat([tf.truncated_normal(shape=self.decoder_input_state.shape),
                                                          self.decoder_input_state],axis = -1)
                    elif Use_VAE:
                        self.decoder_input_state_2 = tf.convert_to_tensor(self.decoder_input_state_2)
                        self.decoder_input_state_2 = tf.transpose(self.decoder_input_state_2,
                                             perm=[2, 0, 1, 3])
                        self.decoder_input_state_2 = tf.reshape(self.decoder_input_state_2, [BATCH_SIZE, -1])
                        self.mn = tf.layers.dense(self.decoder_input_state_2, units=NUM_UNITS*2)
                        self.sd = 0.5 * tf.layers.dense(self.decoder_input_state_2, units=NUM_UNITS*2)
                        epsilon = tf.random_normal(tf.stack([tf.shape(self.decoder_input_state_2)[0], NUM_UNITS*2]))
                        self.decoder_input_state_2 = self.mn + tf.multiply(epsilon, tf.exp(self.sd))
                        self.decoder_input_state_2 = tf.reshape(self.decoder_input_state_2, [BATCH_SIZE, mem_slots, -1])




                        self.decoder_input_state = tf.concat([self.decoder_input_state_2,
                                                          self.decoder_input_state],axis = -1)

                    # build a graph for outputting sequential tokens
                    _, _, self.decoder_final_state, self.outputs = control_flow_ops.while_loop(
                        cond=lambda i, _1, _2, _3: i < max_len,
                        body=_pretrain_recurrence,
                        loop_vars=(tf.constant(1, dtype=tf.int32), tf.nn.embedding_lookup(self.embedding, self.decoder_inputs[:,0]),
                                   self.decoder_input_state, g_predictions))

                    self.logits = tf.transpose(self.outputs.stack()[1:,:,:],
                                                 perm=[1, 0, 2])
                else:
                    self.outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                        cell = self.decoder_cell_drop,
                        initial_state = self.decoder_input_state,#初始状态,h
                        inputs = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs),#输入x
                        sequence_length = self.decoder_lengths,
                        dtype = tf.float32,
                        scope='decoder')



            #这里直接用softmax

            if Use_relational_memory:
                pass
            else:
                # self.logits = g_output_unit(tf.reshape(self.outputs, [-1, _NUM_UNITS]))
                with tf.variable_scope('cell/postprocessing', initializer=xavier_initializer()):
                    self.softmax_w = tf.get_variable('softmax_w', [_NUM_UNITS, param.VOCAB_SIZE])
                    self.softmax_b = tf.get_variable('softmax_b', [param.VOCAB_SIZE])
                self.logits = tf.nn.bias_add(tf.matmul(tf.reshape(self.outputs, [-1, _NUM_UNITS]), self.softmax_w),
                        bias = self.softmax_b)
            self.probs = tf.reshape(tf.nn.softmax(self.logits), [BATCH_SIZE,-1,param.VOCAB_SIZE])#输出应该是一个one_shot向量
            self.labels = tf.one_hot(decoder_inputs[:,1:], depth = param.VOCAB_SIZE,dtype=tf.int32)
            self.right_count = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.one_hot(tf.argmax(self.probs,-1),param.VOCAB_SIZE),
                                                                       tf.to_float(self.labels)),-1),-1)/tf.to_float(max_len)


            self.logits = tf.reshape(self.logits, [BATCH_SIZE,-1,param.VOCAB_SIZE])
            loss = get_loss(LOSS_TYPE,self.logits,self.labels,self.decoder_lengths,BATCH_SIZE)
            self.loss = loss
            self.original_loss = get_loss(1,self.logits,self.labels,self.decoder_lengths,BATCH_SIZE)
            # ---------- generate tokens and approximated one-hot results (Adversarial) ---------
            gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=True)#the prob
            gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, infer_shape=True) # sampled token
            gen_x_onehot_adv = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True,
                                                            infer_shape=True)  # generator output (relaxed of gen_x)

            random_start_length = tf.constant(param.start_length,dtype=tf.int32)

#            random_start_length = tf.random_uniform(shape = [],minval=0,maxval=sentence_min_len,dtype=tf.int32)
            def _start_recurrence(word_i,gen_o,gen_x,gen_x_onehot_adv):
                gen_x = gen_x.write(word_i, decoder_inputs[:, word_i])
                gen_o = gen_o.write(word_i, tf.one_hot(decoder_inputs[:, word_i], param.VOCAB_SIZE, 1.0, 0.0))
                gen_x_onehot_adv = gen_x_onehot_adv.write(word_i,tf.one_hot(decoder_inputs[:, word_i], param.VOCAB_SIZE, 1.0,
                                                                     0.0)*1000000)
                return word_i + 1, gen_o, gen_x, gen_x_onehot_adv
            _, gen_o, gen_x, gen_x_onehot_adv = control_flow_ops.while_loop(
                cond=lambda i,_1,_2,_3: i < random_start_length + 1,
                body=_start_recurrence,
                loop_vars=(tf.constant(0),gen_o, gen_x, gen_x_onehot_adv))

            #temperature = param.temperature
            # the generator recurrent module used for adversarial training
            if Use_relational_memory:
                with tf.variable_scope('cell/postprocessing', reuse = True):
                    if Post_with_state:
                        self.softmax_w = tf.get_variable('softmax_w', [param.head_size*param.num_heads + 2*NUM_UNITS, param.VOCAB_SIZE])
                    else:
                        self.softmax_w = tf.get_variable('softmax_w', [param.head_size*param.num_heads, param.VOCAB_SIZE])

                    self.softmax_b = tf.get_variable('softmax_b', [param.VOCAB_SIZE])
                if param.start_length >= 1:
                    x_emb = tf.transpose(tf.nn.embedding_lookup(self.embedding, self.decoder_inputs),
                                         perm=[1, 0, 2])
                    ta_emb_gen_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=max_len)
                    ta_emb_gen_x = ta_emb_gen_x.unstack(x_emb)
                    # the generator recurrent moddule used for pre-training
                    def _start_rel_recurrence(i, x_t, h_tm1):
                        mem_o_t, h_t = gen_mem(x_t, h_tm1)
                        if Post_with_state:
                            mem_o_t = tf.concat([tf.reshape(mem_o_t, [-1, head_size * num_heads]),
                                                 self.decoder_input_state[:,0,:2*NUM_UNITS]],-1)
                        else:
                            mem_o_t = tf.reshape(mem_o_t, [-1, head_size * num_heads])
                        o_t = tf.nn.bias_add(
                            tf.matmul(mem_o_t, self.softmax_w),
                            bias=self.softmax_b)
                        x_tp1 = ta_emb_gen_x.read(i)
                        return i + 1, x_tp1, h_t

                    self.decoder_start_word_state = \
                        tf.cond(random_start_length > 0,
                        lambda: (control_flow_ops.while_loop(
                        cond=lambda i, _1, _2: i < random_start_length + 1,
                        body=_start_rel_recurrence,
                        loop_vars=(
                        tf.constant(1, dtype=tf.int32),
                        tf.nn.embedding_lookup(self.embedding, self.decoder_inputs[:,0]),
                        self.decoder_input_state)))[2],
                        lambda: self.decoder_input_state)
                    # _, _, self.decoder_start_word_state = control_flow_ops.while_loop(
                    #     cond=lambda i, _1, _2: i < random_start_length + 1,
                    #     body=_start_rel_recurrence,
                    #     loop_vars=(
                    #     tf.constant(1, dtype=tf.int32),
                    #     tf.nn.embedding_lookup(self.embedding, self.decoder_inputs[:,0]),
                    #     self.decoder_input_state))
                else:
                    self.decoder_start_word_state = self.decoder_input_state
            else:
                with tf.variable_scope('cell', reuse= True):
                    _, self.decoder_start_word_state = tf.nn.dynamic_rnn(
                        cell=self.decoder_cell_drop,
                        initial_state=self.encoder_final_state,  # 初始状态,h
                        inputs=tf.nn.embedding_lookup(self.embedding, self.decoder_inputs),  # 输入x
                        sequence_length=np.ones(BATCH_SIZE) * random_start_length,
                        dtype=tf.float32,
                        scope='decoder')

            def _gen_recurrence(i, x_t, state, gen_o, gen_x, gen_x_onehot_adv):
                if Use_relational_memory:
                    mem_o_t, state = gen_mem(x_t, state)  # hidden_memory_tuple
                    if Post_with_state:
                        mem_o_t = tf.concat([tf.reshape(mem_o_t, [-1, head_size * num_heads]),
                                             self.decoder_input_state[:, 0, :2 * NUM_UNITS]],-1)
                    else:
                        mem_o_t = tf.reshape(mem_o_t, [-1, head_size * num_heads])
                    logits = tf.nn.bias_add(
                        tf.matmul(mem_o_t, self.softmax_w),
                        bias=self.softmax_b)

                    pad = np.ones((BATCH_SIZE, VOCAB_SIZE))
                    pad_2 = tf.one_hot(tf.to_int32(tf.constant(np.ones((BATCH_SIZE))*2)),depth=VOCAB_SIZE)
                    pad_2 = tf.multiply(pad_2,end_rate -1)
                    pad = tf.constant(pad)
                    pad = tf.add(tf.to_float(pad),tf.to_float(pad_2))
                    logits = tf.multiply(logits, pad)

                else:
                    with tf.variable_scope('cell', reuse=True):
                        outputs, state = rnn.static_rnn(cell = self.decoder_cell_drop,
                            initial_state = state,
                            inputs = [x_t],#输入x
                            sequence_length = np.ones(BATCH_SIZE),
                            dtype = tf.float32,
                            scope='decoder')
                    with tf.variable_scope('postprocessing', reuse=True):
                        logits = tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, _NUM_UNITS]), self.softmax_w),
                                                bias=self.softmax_b)
#                logits = g_output_unit(tf.reshape(outputs, [-1, _NUM_UNITS]))
                prob = tf.reshape(tf.nn.softmax(logits), [BATCH_SIZE, param.VOCAB_SIZE])#without length
                if not if_test:
                    gumbel_t = add_gumbel(logits)
                else:
#                    gumbel_t = logits
                    gumbel_t = add_gumbel(tf.multiply(1.2, logits))

                next_token = tf.to_int32(tf.stop_gradient(tf.argmax(gumbel_t, axis=1)))
                next_token_onehot = tf.one_hot(next_token, param.VOCAB_SIZE, 1.0, 0.0)
                x_onehot_appr = tf.multiply(gumbel_t, temperature)# one-hot-like, [batch_size x vocab_size]

                gen_o = gen_o.write(i, logits)
                gen_x = gen_x.write(i, next_token)
                gen_x_onehot_adv = gen_x_onehot_adv.write(i, tf.nn.softmax(x_onehot_appr))
                x_tp1 = tf.nn.embedding_lookup(self.embedding, next_token)
                return i + 1, x_tp1, state, gen_o, gen_x, gen_x_onehot_adv
            # build a graph for outputting sequential tokens

            _, _ , _, self.gen_o, self.gen_x, self.gen_x_onehot_adv = control_flow_ops.while_loop(
                cond=lambda i,_1,_2,_3,_4,_5: i < max_len,
                body=_gen_recurrence,
                loop_vars=(random_start_length + 1, tf.nn.embedding_lookup(self.embedding, decoder_inputs[:, random_start_length]),
                           self.decoder_start_word_state, gen_o, gen_x, gen_x_onehot_adv))
            self.gen_o = tf.transpose(self.gen_o.stack(), perm=[1, 0, 2])  # batch_size x seq_len x vocab_size
            self.gen_x = tf.transpose(self.gen_x.stack(), perm=[1, 0])
            self.gen_x_onehot_adv = tf.transpose(self.gen_x_onehot_adv.stack(), perm=[1, 0, 2])

            temp_list = tf.constant(list(range(100)))
            temp_list = temp_list[:max_len] -1
            temp_list = tf.tile(tf.reshape(temp_list,[1,max_len]),[BATCH_SIZE,1])
            self.ifequal = tf.equal(tf.to_int32(tf.argmax(self.gen_o,-1)),tf.ones_like(self.gen_x))
            self.ifequal = tf.to_int32(self.ifequal[:,:-1])
            self.ifequal = tf.concat([self.ifequal,tf.ones((BATCH_SIZE,1),tf.int32)],-1)
            self.total_length = tf.multiply(tf.to_int32(self.ifequal),temp_list) + 10000 * (1-tf.to_int32(self.ifequal))
            self.gen_x_length = tf.reduce_mean(tf.to_float(tf.reduce_min(self.total_length,-1)))


    def init_vars(self, sess):
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

    def pretrain_var_list(self,mean_loss = True):
        if mean_loss:
            return self.decoder_final_state, tf.reduce_mean(self.loss),tf.reduce_mean(self.original_loss)
        else:
            return self.decoder_final_state, tf.reduce_mean(self.loss,1),tf.reduce_mean(self.original_loss)

    def generate_var_list(self):
        return self.gen_o, self.gen_x, self.gen_x_onehot_adv

    def pretrain_logits(self):
        return self.logits
    def encoder_state_var(self):
        return tf.reshape(tf.transpose(tf.convert_to_tensor(self.encoder_final_state),
                                                     perm=[2, 0, 1, 3]),
                                                   [BATCH_SIZE, mem_slots, -1])
    def right_word_count(self):
        return tf.reduce_mean(self.right_count),self.gen_x_length

    def VAE_variable(self):
        return self.mn,self.sd
def prepare_feature_batch(feature_batch, songnames=None, feature_dict=None):
    feature_length = []
    BATCH_SIZE = len(feature_batch)
    for i in range(BATCH_SIZE):
        feature_batch[i] = np.reshape(feature_batch[i][:], [-1])
        temp_length = feature_batch[i].shape[0] // (_NUM_UNITS) * (_NUM_UNITS)
        feature_batch[i] = np.reshape(feature_batch[i][:temp_length*_NUM_UNITS], [-1, _NUM_UNITS])
        feature_length.append(feature_batch[i].shape[0])
    temp_max_length = np.max(feature_length)
    temp_feature_batch = np.zeros([BATCH_SIZE, temp_max_length, _NUM_UNITS])
    for i in range(BATCH_SIZE):
        temp_feature_batch[i, 0:feature_batch[i].shape[0], :] = feature_batch[i]
    feature_batch = temp_feature_batch
    feature_length = np.array(feature_length)

    song_names_dict = dict()
    song_name_list = list(feature_dict.keys())
    for songidx,song_name in enumerate(song_name_list):
        song_names_dict[song_name] = songidx
    song_name_labels = []
    for i in range(BATCH_SIZE):
        song_name_labels.append(song_names_dict[songnames[i]])
    song_name_labels = np.array(song_name_labels)
    return feature_batch,feature_length,song_name_labels

def prepare_lyric_batch(lyric_batch,ch2int):
    lyric_length = []
    temp_lyric_batch  = []
    for i in range(BATCH_SIZE):
        temp_length = len(lyric_batch[i])
        lyric_length.append(temp_length)
        temp_lyric_batch.append([])
    temp_max_length = np.max(lyric_length)
    temp_lyric_batch = np.zeros([BATCH_SIZE, temp_max_length], dtype=np.int32)
    for i in range(BATCH_SIZE):
        for j in range(lyric_length[i]):
            try:
                temp_lyric_batch[i][j] = ch2int[lyric_batch[i][j]]
            except:
                pass
    lyric_length = np.array(lyric_length)
    return temp_lyric_batch,lyric_length

