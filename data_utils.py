import numpy as np
import random
import tensorflow as tf
import os
import sys
from param import *
import param
from vocab import get_vocab,get_corpus,text_preprocess
_NUM_UNITS = param.NUM_UNITS#input width of LSTM
# 不知道为什么作为class Generator 的函数就会出错..
encoder_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, None, _NUM_UNITS])
encoder_lengths = tf.placeholder(tf.int32, [BATCH_SIZE])
decoder_lengths = tf.placeholder(tf.int32, [BATCH_SIZE])
decoder_inputs = tf.placeholder(tf.int32, [BATCH_SIZE, None])
encoder_inputs_2 = tf.placeholder(tf.int32, [BATCH_SIZE, None])
encoder_lengths_2 = tf.placeholder(tf.int32, [BATCH_SIZE])
embed_ph = tf.placeholder(tf.float32, [param.VOCAB_SIZE, INPUT_DIM])
song_label = tf.placeholder(tf.int32, [BATCH_SIZE])

class DataLoader:
    def __init__(self):
        self.int2ch, self.ch2int = get_vocab(if_segment)

    def get_batches(self, batch_size,set_no,feature_dict = None,lyric_dict = None,songidx = None,num = None):
        if set_no == 4:
            total_text, song_names = get_corpus(if_segment=if_segment, set_no=3,songidx = songidx)
        else:
            total_text,song_names = get_corpus(if_segment= if_segment,set_no = set_no,songidx = songidx)

        n_batches = (len(total_text) // (batch_size))
        if num is not None:
            n_batches = num
        batches = []
        length = []
        features = []
        songnames = []
        lyrics = []
        total_length = len(total_text)
        if set_no <= 3:# random.shuffle
            temp_rand_num = np.random.randint(1, 1000000, total_length)
            temp_rank = np.argsort(temp_rand_num)#打乱
            new_text = []
            new_names = []
            for i in temp_rank:
                new_text.append(total_text[i])
                new_names.append(song_names[i])
            total_text = new_text
            song_names = new_names
        for text_i,text in enumerate(total_text):
            for word_i,word in enumerate(text):
                total_text[text_i][word_i] = self.ch2int[total_text[text_i][word_i]]
        max_length = []
        global_max_length = 0
        for i in range(total_length):
            global_max_length = max(global_max_length,total_text[i].__len__())
        for batch_i in range(n_batches):
            batch_length = []
            for j in range(batch_size):
                batch_length.append(len(total_text[batch_i * batch_size + j]))
            max_length.append(max(batch_length))
            length.append(batch_length)
            for text_i,text in enumerate(total_text[batch_i*batch_size : (batch_i+1)*batch_size]):
                text_length = len(text)
                for zero_i in range(text_length ,global_max_length):
                    total_text[batch_i * batch_size + text_i].append(self.ch2int['\end'])#调节为等长度，以便化为array
            batches.append(np.array(total_text[batch_i * batch_size:(batch_i + 1) * (batch_size)]))
            if feature_dict is not None:
                feature = []
                feature_per_song = 2000
                for j in range(batch_size):
                    # feature_length = len(feature_dict[song_names[batch_i * batch_size + j]])
    #                start = random.randint(0, feature_length//3 * 1)
    #                end = random.randint(feature_length//3 * 2,feature_length)
                    if param.divide_type == 2:
                        if set_no == 0:
                            feature_idx = np.random.randint(0, feature_per_song, 1)
                        elif set_no == 1:
                            feature_idx = np.random.randint(0, feature_per_song, 1)
                        elif set_no == 2:
                            feature_idx = np.random.randint(feature_per_song, feature_per_song + 50, 1)
                        elif set_no == 3:
                            feature_idx = np.random.randint(feature_per_song, feature_per_song + 50, 1)
                        elif set_no == 4:
                            feature_idx = np.random.randint(feature_per_song, feature_per_song + 50, 1)
                    else:
                        feature_idx = np.random.randint(0,feature_per_song,1)

                    feature.append(feature_dict[song_names[batch_i * batch_size + j]][feature_idx])
                features.append(feature)
            if lyric_dict is not None:
                lyric = []
                for j in range(batch_size):
                    lyric_num = len(lyric_dict[song_names[batch_i * batch_size + j]])
                    lyric_idx = np.random.randint(0,lyric_num , 1)[0]
                    lyric.append(lyric_dict[song_names[batch_i * batch_size + j]][lyric_idx])
                lyrics.append(lyric)
            temp_name = []
            for j in range(batch_size):
                temp_name.append(song_names[batch_i * batch_size + j])
            songnames.append(temp_name)
        return batches,length,features,songnames,lyrics

    def get_target_batch(self,batch,length):
        target_batch = batch.copy()
        for i in range(BATCH_SIZE):
            for j in range(length[i], batch.shape[-1]):
                target_batch[i][j] = self.ch2int['\end']
        return target_batch

    def get_audio_feature(self):
        files = os.listdir(get_data_dir())
        feature = dict()
        for file in files:
            try:
                temp = np.load(get_data_dir() + '/' + file + '/' + file + '.npy')
                feature[file] = temp
            except:
                pass
        # print('audio feature read')
        return feature

    def get_lyric(self,):
        files = os.listdir(get_data_dir())
        lyric = dict()
        for file in files:
            try:
                f = open(get_data_dir()+'/'+file+'/'+file+'_lyric.txt', 'r',encoding = 'utf-8')  # 打开文件
                data = f.readlines()
                data = text_preprocess(data)
                lyric[file] = data
            except:
                pass
        return lyric

def is_chinese(word):
    for uchar in word:
        """判断一个unicode是否是汉字"""
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
            return True
    else:
        return False


def process_for_bleu(batches,ch2int):
    temp_valid_batches = []
    for batch in batches:
        new_batch = batch.tolist()
        for sen in new_batch:
            temp_valid_batches.append(rm_end(sen, ch2int))
    batches = temp_valid_batches
    return batches
def rm_end(sentence,ch2int):
    for id,ch in enumerate(sentence):
        if ch == ch2int['\end']:
            return sentence[:id + 1]
    new_sen = sentence
    new_sen[-1] = ch2int['\end']
    return new_sen


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')

def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None

def get_loss(loss_type,logits,labels,decoder_lengths,BATCH_SIZE):
    temp_len_labels = []
    temp_len_logits = []
    for i in range(BATCH_SIZE):
        temp_len_labels.append(labels[0, 0:decoder_lengths[i], :])
        temp_len_logits.append(logits[0, 0:decoder_lengths[i], :])
    logits_2 = tf.concat(temp_len_logits, 0)
    labels_2 = tf.concat(temp_len_labels, 0)

    if loss_type == 1:
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits_2,
            labels=labels_2)
    elif loss_type == 2:
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels)
    return loss

def get_pretrain_op(loss,learn_rate):

    var_list_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator/embedding')  # for the embedding matrix
    var_list_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator/cell') + \
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator/postprocessing')  # for the RNN
    opt_op_1 = tf.train.AdamOptimizer(learn_rate/100)
    opt_op_2 = tf.train.AdamOptimizer(learn_rate)
    # 裁剪一下Gradient输出，最后的gradient都在[-1, 1]的范围内
    gradients_1 = opt_op_1.compute_gradients(loss, var_list=var_list_1)
    gradients_2 = opt_op_2.compute_gradients(loss, var_list=var_list_2)
    # 修建gradient
    capped_gradients_1 = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gradients_1 if
                          grad is not None]
    capped_gradients_2 = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gradients_2 if
                          grad is not None]
    train_op_1 = opt_op_1.apply_gradients(capped_gradients_1)
    train_op_2 = opt_op_2.apply_gradients(capped_gradients_2)
    train_op = tf.group(train_op_1, train_op_2)
    return train_op
def get_reco_op(state,song_label,lr):

    with tf.variable_scope('postprocessing'):
        softmax_w = tf.get_variable('softmax_w', [_NUM_UNITS * 2, Songnum])
        softmax_b = tf.get_variable('softmax_b', [Songnum])
    logits = tf.nn.bias_add(tf.matmul(tf.reshape(state, [-1, _NUM_UNITS * 2]), softmax_w),
                            bias=softmax_b)
    probs = tf.reshape(tf.nn.softmax(logits), [BATCH_SIZE, -1, Songnum])  # 输出应该是一个one_shot向量
    song_label = tf.one_hot(song_label, depth=Songnum, dtype=tf.int32)
    logits = tf.reshape(logits, [BATCH_SIZE, Songnum])
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=song_label)
    loss = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(lr)
    gradients = opt.compute_gradients(loss)
    optim_2 = opt.apply_gradients(gradients)
    return optim_2,loss
def add_gumbel(o_t, eps=1e-10):
    """Sample from Gumbel(0, 1)"""
    u = tf.random_uniform(tf.shape(o_t), minval=0, maxval=1, dtype=tf.float32)
    g_t = -tf.log(-tf.log(u + eps) + eps)
    gumbel_t = tf.add(o_t, g_t)
    return gumbel_t

def conv2d(input_, out_nums, k_h=2, k_w=1, d_h=2, d_w=1, padding='SAME', scope=None):
    in_nums = input_.get_shape().as_list()[-1]
    # Glorot initialization
    with tf.variable_scope(scope or "Conv2d"):
        W = tf.get_variable("Matrix", shape=[k_h, k_w, in_nums, out_nums],
                            initializer=tf.truncated_normal_initializer())
        # if sn:
        #     W = spectral_norm(W)
        b = tf.get_variable("Bias", shape=[out_nums], initializer=tf.zeros_initializer)
        conv = tf.nn.conv2d(input_, filter=W, strides=[1, d_h, d_w, 1], padding=padding)
        conv = tf.nn.bias_add(conv, b)

    return conv
def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output
def linear(input_, output_size, use_bias=False, sn=False, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: Variable Scope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        W = tf.get_variable("Matrix", shape=[output_size, input_size],
                            # initializer=create_linear_initializer(input_size, input_.dtype),
                            dtype=input_.dtype)
        # if sn:
        #     W = spectral_norm(W)
        output_ = tf.matmul(input_, tf.transpose(W))
        if use_bias:
            bias_term = tf.get_variable("Bias", [output_size],
                                        # initializer=create_bias_initializer(input_.dtype),
                                        dtype=input_.dtype)
            output_ += bias_term

    return output_


def get_losses(d_out_real, d_out_fake,gan_type = 'RSGAN'):
    if gan_type == 'standard':  # the non-satuating GAN loss
        noise = tf.random_uniform(shape=d_out_real.shape) * 0.1
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real) - noise
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake) + noise
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.ones_like(d_out_fake)
        ))

    elif gan_type == 'JS':  # the vanilla GAN loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -d_loss_fake

    elif gan_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(-d_out_fake)

    elif gan_type == 'hinge':  # the hinge loss
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_out_real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -tf.reduce_mean(d_out_fake)

    elif gan_type == 'tv':  # the total variation distance
        d_loss = tf.reduce_mean(tf.tanh(d_out_fake) - tf.tanh(d_out_real))
        g_loss = tf.reduce_mean(-tf.tanh(d_out_fake))

    # elif gan_type == 'wgan-gp': # WGAN-GP
    #     d_loss = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
    #     GP = gradient_penalty(discriminator, x_real_onehot, x_fake_onehot_appr, config)
    #     d_loss += GP
    #
    #     g_loss = -tf.reduce_mean(d_out_fake)

    elif gan_type == 'LS': # LS-GAN
        d_loss_real = tf.reduce_mean(tf.squared_difference(d_out_real, 1.0))
        d_loss_fake = tf.reduce_mean(tf.square(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(tf.squared_difference(d_out_fake, 1.0))

    elif gan_type == 'RSGAN':  # relativistic standard GAN
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real - d_out_fake, labels=tf.ones_like(d_out_real)
        ))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake - d_out_real, labels=tf.ones_like(d_out_fake)
        ))

    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % gan_type)

    # log_pg = tf.reduce_mean(tf.log(gen_o + EPS))  # [1], measures the log p_g(x)

    return g_loss, d_loss


def get_train_ops(g_loss = None, d_loss_1 = None,d_loss_2 = None, temperature = 0, global_step = 0,learn_rate = 0.01):
    optimizer_name = 'adam'
    nadv_steps = 5
    d_lr = learn_rate * d_lr_rate
    gpre_lr = learn_rate
    gadv_lr = learn_rate * g_lr_rate
    if g_loss is not None:
        g_vars_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator/embedding')  # for the embedding matrix
        g_vars_2= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator/cell') + \
                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator/postprocessing')
#    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    if d_loss_1 is not None:
        d_vars_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    if d_loss_2 is not None:
        d_vars_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_2')\
               + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    grad_clip = 5.0  # keep the same with the previous setting

    # generator pre-training
    # pretrain_opt = tf.train.AdamOptimizer(gpre_lr, beta1=0.9, beta2=0.999)
    # pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(g_pretrain_loss, g_vars), grad_clip)  # gradient clipping
    # g_pretrain_op = pretrain_opt.apply_gradients(zip(pretrain_grad, g_vars))

    # d_lr = tf.train.exponential_decay(d_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)
    # gadv_lr = tf.train.exponential_decay(gadv_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)

    if optimizer_name == 'adam':
        d_optimizer = tf.train.AdamOptimizer(d_lr, beta1=0.9, beta2=0.999)
        g_optimizer_1 = tf.train.AdamOptimizer(gadv_lr/100, beta1=0.9, beta2=0.999)
        g_optimizer_2 = tf.train.AdamOptimizer(gadv_lr, beta1=0.9, beta2=0.999)
#        temp_optimizer = tf.train.AdamOptimizer(1e-2, beta1=0.9, beta2=0.999)
    elif optimizer_name == 'rmsprop':
        d_optimizer = tf.train.RMSPropOptimizer(d_lr)
        g_optimizer = tf.train.RMSPropOptimizer(gadv_lr)
    else:
        raise NotImplementedError

    # g_grads, _ = tf.clip_by_global_norm(tf.gradients(g_loss, g_vars), grad_clip)  # gradient clipping
    # g_train_op = g_optimizer.apply_gradients(zip(g_grads, g_vars))
    g_train_op, d_train_op_1, d_train_op_2 = 0,0,0
    if g_loss is not None:
        g_grads_1, _ = tf.clip_by_global_norm(tf.gradients(g_loss, g_vars_1), grad_clip)
        g_train_op_1 = g_optimizer_1.apply_gradients(zip(g_grads_1, g_vars_1))
        g_grads_2, _ = tf.clip_by_global_norm(tf.gradients(g_loss, g_vars_2), grad_clip)
        g_train_op_2 = g_optimizer_2.apply_gradients(zip(g_grads_2, g_vars_2))
        g_train_op = tf.group(g_train_op_1,g_train_op_2)

        print('len of g_grads without None: {}'.format(len([i for i in g_grads_2 if i is not None])))
        print('len of g_grads: {}'.format(len(g_grads_2)))

    # d_train_op = d_optimizer.minimize(d_loss, var_list=d_vars)
    if d_loss_1 is not None:
        d_grads_1, _ = tf.clip_by_global_norm(tf.gradients(d_loss_1, d_vars_1), grad_clip)  # gradient clipping
        d_train_op_1 = d_optimizer.apply_gradients(zip(d_grads_1, d_vars_1))
    if d_loss_2 is not None:
        d_grads_2, _ = tf.clip_by_global_norm(tf.gradients(d_loss_2 * d_2_rate, d_vars_2), grad_clip)  # gradient clipping
        d_train_op_2 = d_optimizer.apply_gradients(zip(d_grads_2, d_vars_2))

    return g_train_op, d_train_op_1,d_train_op_2


def find_uninitialized_var(sess):
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    return uninitialized_vars
