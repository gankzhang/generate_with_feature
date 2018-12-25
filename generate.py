from vocab import *
#from tensorflow.contrib import seq2seq
from nltk.translate.bleu_score import sentence_bleu
import tensorflow as tf
from audio_feature import *
from tensorflow.contrib import rnn
import numpy as np
import random
import argparse
import sys
import time
from pick_word import *
_model_path = os.path.join(param.get_data_dir(), 'model')
data_dir = param.get_data_dir()
save_dir = param.get_save_dir()
_NUM_UNITS = param.get_embed_dim()#输入宽度
_NUM_LAYERS = 2
BATCH_SIZE = 20
if_segment = param.if_segment()

stop_dir = param.get_stop_dir()
stopwords = [line.strip() for line in (open(stop_dir, 'r', encoding='utf-8')).readlines()]
stopwords.append(' ')
stopwords.append('\n')

stopwords_dict = dict()
for stopword in stopwords:
    stopwords_dict[stopword] = 1


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--lr', type=float, default=0.01,help = 'learning rate')
    parser.add_argument('--decay_rate',type=float,default=0.9,help = 'lr decay rate')
    parser.add_argument('--batch_size',type = int,default=16,help = 'batch size')
    parser.add_argument('--replace_decay',type=float,default=1,help = 'the decay of the scheduled sampling')
    parser.add_argument('--num_layer',type = int,default=_NUM_LAYERS,help = 'the num of layers of the RNN')
    parser.add_argument('--n_epoch',type = int,default = 50,help= 'the num of the training epoch')
    return parser.parse_args()


class Generator:
    def __init__(self):
        keep_prob = 0.8# dropout
        self.int2ch, self.ch2int = get_vocab(if_segment)
        VOCAB_SIZE = len(self.int2ch)
        self.learn_rate = tf.Variable(0.0, trainable = False)
        with tf.variable_scope('cell',initializer=tf.contrib.layers.xavier_initializer()):
            self.encoder_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(_NUM_UNITS)] * _NUM_LAYERS)
            self.encoder_init_state = self.encoder_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
            self.encoder_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, None,_NUM_UNITS])
            self.encoder_lengths = tf.placeholder(tf.int32, [BATCH_SIZE])
            _, self.encoder_final_state = tf.nn.dynamic_rnn(
                cell=self.encoder_cell,
                initial_state=self.encoder_init_state,
                inputs=self.encoder_inputs,
                sequence_length=self.encoder_lengths,
                scope='encoder')

            # embedding 层
            self.embedding = tf.get_variable(name = 'embedding',shape=[VOCAB_SIZE, _NUM_UNITS],trainable = True,
                                             initializer=tf.contrib.layers.xavier_initializer())
            self._embed_ph = tf.placeholder(tf.float32, [VOCAB_SIZE, _NUM_UNITS])
            self._embed_init = self.embedding.assign(self._embed_ph)

#            self.decoder_cell = rnn.MultiRNNCell([rnn.GRUCell(_NUM_UNITS)] * _NUM_LAYERS)
#            self.initial_state = tf.placeholder(tf.float32, [_NUM_LAYERS * _NUM_UNITS * 2])
            self.decoder_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(_NUM_UNITS)] * _NUM_LAYERS)
            self.decoder_cell_drop = tf.contrib.rnn.DropoutWrapper(self.decoder_cell, output_keep_prob=keep_prob)
#            self.decoder_init_state = self.decoder_cell_drop.zero_state(BATCH_SIZE,dtype=tf.float32)
#            self.decoder_init_state = self.decoder_cell_drop.get_initial_state(self.initial_state,BATCH_SIZE,dtype=tf.float32)

            self.decoder_lengths = tf.placeholder(tf.int32, [BATCH_SIZE])
            self.decoder_inputs = tf.placeholder(tf.int32, [BATCH_SIZE, None])
            self.outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                cell = self.decoder_cell_drop,
                initial_state = self.encoder_final_state,#初始状态,h
                inputs = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs),#输入x
                sequence_length = self.decoder_lengths,
                dtype = tf.float32,
                scope = 'decoder')

        #这里直接用softmax
        with tf.variable_scope('decoder'):
            self.softmax_w = tf.get_variable('softmax_w', [_NUM_UNITS, VOCAB_SIZE],initializer=tf.contrib.layers.xavier_initializer())
            self.softmax_b = tf.get_variable('softmax_b', [VOCAB_SIZE],initializer=tf.contrib.layers.xavier_initializer())

        self.logits = tf.nn.bias_add(tf.matmul(tf.reshape(self.outputs, [-1, _NUM_UNITS]), self.softmax_w),
                bias = self.softmax_b)
        self.probs = tf.reshape(tf.nn.softmax(self.logits), [BATCH_SIZE,-1,VOCAB_SIZE])#输出应该是一个one_shot向量
        self.targets = tf.placeholder(tf.int32, [BATCH_SIZE, None])#训练用
        self.labels = tf.one_hot(self.targets, depth = VOCAB_SIZE,dtype=tf.int32)
        self.logits = tf.reshape(self.logits, [BATCH_SIZE,-1,VOCAB_SIZE])

        temp_len_labels = []
        temp_len_logits = []
        for i in range(BATCH_SIZE):
            temp_len_labels.append(self.labels[0,0:self.decoder_lengths[i],:])
            temp_len_logits.append(self.logits[0, 0:self.decoder_lengths[i], :])
        self.logits_2 = tf.concat(temp_len_logits,0)
        self.labels = tf.concat(temp_len_labels,0)

        loss = tf.nn.softmax_cross_entropy_with_logits(
                logits = self.logits_2,
                labels = self.labels)
        self.loss = tf.reduce_mean(loss)

        # 裁剪一下Gradient输出，最后的gradient都在[-1, 1]的范围内
        self.opt_op = tf.train.AdamOptimizer(self.learn_rate)
        self.gradients = self.opt_op.compute_gradients(self.loss)#TODO : add var_list for different lr
        # 修建gradient
        self.capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gradients if grad is not None]
        self.train_op = self.opt_op.apply_gradients(self.capped_gradients)



    def _init_vars(self, sess):
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)


    def get_batches(self, batch_size,set_no,padding_num = 0,feature_dict = dict()):
        if set_no == 4:
            total_text, song_names = get_corpus(if_segment=if_segment, set_no=1)
            total_text = total_text[0:1000]
            song_names = song_names[0:1000]
        else:
            total_text,song_names = get_corpus(if_segment= if_segment,set_no = set_no)

        n_batches = (len(total_text) // (batch_size))
        batches = []
        length = []
        features = []
        songnames = []

        if set_no != 4:# random.shuffle
            total_length = len(total_text)
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
        for batch_i in range(n_batches):
            batch_length = []
            for j in range(batch_size):
                batch_length.append(len(total_text[batch_i * batch_size + j]))
            max_length.append(max(batch_length))
            length.append(batch_length)

            for text_i,text in enumerate(total_text[batch_i*batch_size : (batch_i+1)*batch_size]):
                text_length = len(text)
                for zero_i in range(text_length ,max_length[batch_i]):
                    total_text[batch_i * batch_size + text_i].append(3)#调节为等长度，以便化为array
                    #
            batches.append(np.array(total_text[batch_i * batch_size:(batch_i + 1) * (batch_size)]))

            temp_feature = []
            for j in range(batch_size):
                temp_feature.append(feature_dict[song_names[batch_i * batch_size + j]])
            features.append(temp_feature)
            temp_name = []
            for j in range(batch_size):
                temp_name.append(song_names[batch_i * batch_size + j])
            songnames.append(temp_name)
        return batches,length,features,songnames

    def get_target_batch(self,batch,length):
        target_batch = batch.copy()
        for i in range(BATCH_SIZE):
            for j in range(length[i], batch.shape[-1]):
                target_batch[i][j] = 3
        return target_batch
    def train(self,n_epochs = 100):
        learn_rate = args.lr
        decay_rate = args.decay_rate
        replace_decay = args.replace_decay
        last_valid_loss = 0
        last_train_loss = 0

        print("Start training RNN enc-dec model ...")
        embedding = np.load(os.path.join(data_dir,'word2vec.npy'))
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("logs/", sess.graph)
            saver = tf.train.Saver()
            self._init_vars(sess)
            saved_global_step = load(saver, sess, save_dir)
            if saved_global_step is None:
                # The first training step will be saved_global_step + 1,
                # therefore we put -1 here for new or overwritten trainings.
                saved_global_step = -1
            try:
                feature_dict = get_audio_feature()
                for epoch in range(saved_global_step + 1,n_epochs):
                    log = open("log.txt", "a+")
                    replace_prob = 1 - replace_decay ** epoch
                    batches, length,features,_ = self.get_batches(batch_size=BATCH_SIZE, set_no=1,feature_dict=feature_dict)
                    sess.run(tf.assign(self.learn_rate, learn_rate * decay_rate ** (epoch//5*5)))
                    total_loss = 0
                    temp_total_loss = 0
                    for batch_i,batch in enumerate(batches[0:50]):
                        replaced_batch = self.replace(sess,batch,length[batch_i],replace_prob)
                        target_batch = self.get_target_batch(batch,length[batch_i])
                        feature_batch = features[batch_i]
                        feature_batch, feature_length = prepare_feature_batch(feature_batch)
                        outputs, loss,_ = sess.run([self.decoder_final_state, self.loss, self.train_op], feed_dict={
                            self.encoder_inputs:feature_batch,
                            self.encoder_lengths:feature_length,
                            self.decoder_inputs:replaced_batch[:,:-1],
                            self.targets: target_batch[:,1:],
                            self.decoder_lengths:length[batch_i],
                            self._embed_ph: embedding})
                        total_loss += loss
                        if batch_i  %  10 == 9:
                            print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f} range_loss = {:.3f}'.format(
                                epoch,
                                batch_i + 1,
                                len(batches),
                                loss,
                                total_loss - temp_total_loss))
                            temp_total_loss = total_loss
                    print('Epoch {:>3}   total_loss = {:.3f}({:.3f}),avg = {:.3f} lr = {:.5f} replace_prob = {:.3f}'.format(
                        epoch,total_loss,total_loss - last_train_loss,total_loss / len(batches),sess.run(self.learn_rate),replace_prob))
                    log.write('Epoch {:>3}   total_loss = {:.3f}({:.3f}),avg = {:.3f} lr = {:.5f} replace_prob = {:.3f}\n'.format(
                        epoch,total_loss,total_loss - last_train_loss,total_loss/len(batches),sess.run(self.learn_rate),replace_prob))
                    last_train_loss = total_loss
                    print(time.ctime())
                    log.write(time.ctime()+'\n')
                    #valid
                    last_valid_loss = self.valid(sess,embedding,log,last_valid_loss)
                    #save
                    if epoch % 5 == 0:
                        save(saver, sess, save_dir, epoch)
                        last_saved_epoch = epoch
                        print('Model Trained and Saved')
#                        self.generate(1)
                    log.close()
            except KeyboardInterrupt:
                print("\nTraining is interrupted.")

    def valid(self,sess,embedding,log,last_valid_loss):
        total_valid_loss = 0
        total_valid_times = 0
        batches, length, features, _ = self.get_batches(BATCH_SIZE, set_no=2, feature_dict=get_audio_feature())
        for batch_i, batch in enumerate(batches[0:50]):
            feature_batch = features[batch_i]
            feature_batch,feature_length = prepare_feature_batch(feature_batch)
            target_batch = self.get_target_batch(batch, length[batch_i])
            outputs, loss = sess.run([self.decoder_final_state, self.loss], feed_dict={
                self.encoder_inputs: feature_batch,
                self.encoder_lengths: feature_length,
                self.decoder_inputs: batch[:, :-1],
                self.targets: target_batch[:, 1:],
                self.decoder_lengths: length[batch_i],
                self._embed_ph: embedding})
            total_valid_loss += loss
            total_valid_times += 1
        print('Valid Total loss = {:.3f}({:.3f}),avg = {:.3f}'.format(
            total_valid_loss,total_valid_loss - last_valid_loss,total_valid_loss / total_valid_times))
        log.write('Valid Total loss = {:.3f}({:.3f}),avg = {:.3f}\n'.format(
            total_valid_loss,total_valid_loss - last_valid_loss,total_valid_loss / total_valid_times))
        last_valid_loss = total_valid_loss
        return last_valid_loss

    def generate(self,num = None,start_word = 5,write_json = False):
        embedding = np.load(os.path.join(data_dir, 'word2vec.npy'))
        with tf.Session() as sess:
            saver = tf.train.Saver()
            self._init_vars(sess)
            _ = load(saver, sess, save_dir)
            batches,lengthes,features,songnames= self.get_batches(BATCH_SIZE , set_no=4,feature_dict = get_audio_feature())
            if num!= None:
                batches = batches[:num]
                lengthes = lengthes[:num]
            if write_json:
                strings = dict()

            for batch_i,batch in enumerate(batches):
                total_length = np.max(lengthes[batch_i])#一个batch中的最大长度
                prime_word = batch[:,0:start_word].tolist()
                gen_sentence = []
                song_names = []
                gen_string = []
                for i in range(BATCH_SIZE):
                    gen_sentence.append([])
                    song_names.append(songnames[batch_i][i])
                    gen_sentence[i].extend(prime_word[i])
                for length in range(start_word - 1, total_length):
                    if length == start_word - 1:#initialize the state
                        input_length = start_word * np.ones(BATCH_SIZE)
                        feature_batch = features[batch_i]
                        feature_batch,feature_length = prepare_feature_batch(feature_batch)

                        prob, state = sess.run([self.probs, self.decoder_final_state], feed_dict={
                            self.encoder_inputs: feature_batch,
                            self.encoder_lengths: feature_length,
                            self.decoder_inputs: np.array(gen_sentence),
                            self.decoder_lengths: input_length,
                            self._embed_ph: embedding})
                        continue
                    else:
                        input_length = np.ones(BATCH_SIZE)
                        prob, state = sess.run([self.probs, self.decoder_final_state], feed_dict={
                            self.decoder_inputs: np.array(gen_sentence)[:,-1:],
                            self.decoder_init_state: state,
                            self.decoder_lengths: input_length,
                            self._embed_ph: embedding})
                    for i in range(BATCH_SIZE):#pick the word
                        temp = prob[i,- 1].tolist()
                        pre_word = pick_word(temp)
                        gen_sentence[i].append(pre_word)
                batch = batch.tolist()

                for i in range(BATCH_SIZE):#为后面的输出batch做处理
                    for word_i in range(total_length):
                        batch[i][word_i] = self.int2ch[batch[i][word_i]]
                        if batch[i][word_i] == '\end':
                            batch[i] = batch[i][0:word_i + 1]
                            break
                for sentence in range(BATCH_SIZE):
                    for word_i in range(total_length):
                        gen_sentence[sentence][word_i] = self.int2ch[gen_sentence[sentence][word_i]]
                        if gen_sentence[sentence][word_i] == '\end' or word_i == (total_length - 1):
                            gen_sentence[sentence][-1] = '\end'
                            print('生成的句子: ',end='')
                            for word_ii in range(word_i + 1):
                                print(gen_sentence[sentence][word_ii],end='')
                            gen_sentence[sentence] = gen_sentence[sentence][0:word_i + 1]
                            string = ''.join(gen_sentence[sentence][0:word_i + 1])
                            gen_string.append(string[1:-1])
                            break
                    print('')

                    print('原来的句子: ', end='')
                    for word in batch[sentence]:
                        print(word,end='')
                    print('')


                if write_json:
                    for sen_i,sen in enumerate(gen_string):
                        try:
                            strings[songnames[batch_i][sen_i]].append(sen)
                        except:
                            strings[songnames[batch_i][sen_i]] = [sen]

    def replace(self,sess,batch,batch_length,replace_prob): #Schedueld Sampling
        if replace_prob < 1e-4:
            return batch
        embedding = np.load(os.path.join(data_dir, 'word2vec.npy'))
        max_len = 0
        for i in range(BATCH_SIZE):
            if batch_length[i] > max_len:
                max_len  = batch_length[i]
        state = sess.run(self.decoder_init_state)
        for length in range(max_len):
            if length < 5:
                continue
#            input_length = length * np.ones(BATCH_SIZE)
            if length == 5:
                input_length = 5 * np.ones(BATCH_SIZE)
                prob, state = sess.run([self.probs, self.decoder_final_state], feed_dict={
                    self.decoder_init_state: state,
                    self.decoder_inputs: batch[:, :length],
                    self.decoder_lengths: input_length,
                    self._embed_ph: embedding})
            else:
                input_length = np.ones(BATCH_SIZE)
                prob, state = sess.run([self.probs, self.decoder_final_state], feed_dict={
                    self.decoder_init_state:state,
                    self.decoder_inputs: batch[:,-1:],
                    self.decoder_lengths: input_length,
                    self._embed_ph: embedding})
            if random.random() < replace_prob:
                for i in range(BATCH_SIZE):#不替换padding
                    temp = prob[i, -1].tolist()
                    pre_word = pick_word(temp)
                    if i<batch_length[i]:
                        batch[i,length] = pre_word
                    else:
                        continue
            else:
                continue
        return batch


# 不知道为什么作为class Generator 的函数就会出错..
def pick_word(probabilities):#return int
    probabilities[3]/=2# \end have less prob
    candidate = np.argsort(probabilities)[-5:]
    new_prob = []
    for ch in candidate:
#        if probabilities[ch] > 0.05:
        new_prob.append(probabilities[ch])
    sums = np.sum(np.array(new_prob))
    for i in range(len(new_prob)):
        new_prob[i] /= sums
    random_num = random.random()
    for i in range(new_prob.__len__()):
        random_num -= new_prob[i]
        if random_num < 0:
            ch = candidate[i]
            break
    if len(candidate) == 0:
        ch = np.argmax(probabilities)
#    if self.ch2int(ch) ==
    return ch

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

def prepare_feature_batch(feature_batch):
    feature_length = []
    for i in range(BATCH_SIZE):
        temp_length = feature_batch[i].shape[0] // (_NUM_UNITS * 10) * (_NUM_UNITS * 10)
        feature_batch[i] = np.reshape(feature_batch[i][:temp_length], [10, -1, _NUM_UNITS])
        feature_batch[i] = np.mean(feature_batch[i], 0)
        feature_length.append(feature_batch[i].shape[0])
    temp_max_length = np.max(feature_length)
    temp_feature_batch = np.zeros([BATCH_SIZE, temp_max_length, _NUM_UNITS])
    for i in range(BATCH_SIZE):
        temp_feature_batch[i, 0:feature_batch[i].shape[0], :] = feature_batch[i]
    feature_batch = temp_feature_batch
    feature_length = np.array(feature_length)
    return feature_batch,feature_length

if __name__ == '__main__':
    args = get_arguments()
    _NUM_LAYERS = args.num_layer
    generator = Generator()
    n_epoch = args.n_epoch
    generator.train(n_epoch)
#    generator.generate(1)