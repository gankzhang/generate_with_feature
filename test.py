from generate import *
_NUM_UNITS = param.NUM_UNITS
import warnings
import scipy
from data_utils import rm_end,process_for_bleu
from discriminator_2 import discriminator_2
warnings.filterwarnings('ignore')
def compare_bleu(num = None,sess = None,write_json = False , generator=None,batch_size = param.BATCH_SIZE,model = 'mle'):

    int2ch, ch2int = get_vocab(if_segment)
    param.VOCAB_SIZE = len(int2ch)
    BATCH_SIZE = batch_size
    _embed_ph = tf.placeholder(tf.float32, [param.VOCAB_SIZE, INPUT_DIM])
    learn_rate = tf.Variable(0.0, trainable=False)
    temperature = tf.Variable(1.0,trainable=False)
    end_rate = tf.Variable(1.0,trainable=False)
    embedding = np.load(os.path.join(data_dir, 'word2vec.npy'))
    dataloader = DataLoader()
    if sess is None:
        sess = tf.Session()
        generator = Generator(encoder_inputs, encoder_lengths, encoder_inputs_2, encoder_lengths_2, decoder_inputs,
                              decoder_lengths , _embed_ph, learn_rate, ch2int['\\start'], if_test=True,
                              temperature = temperature,end_rate = end_rate)
        generator.init_vars(sess)
        gen_o, gen_x, gen_x_onehot_adv = generator.generate_var_list()
        _, pretrain_loss, _ = generator.pretrain_var_list()
        saver = tf.train.Saver()
        if model == 'mle':
            _ = load(saver, sess, param.get_save_dir())
        elif model =='gan':
            _ = load(saver, sess, param.get_adv_save_dir())

    else:
        gen_o, gen_x, gen_x_onehot_adv = generator.generate_var_list()
        _, pretrain_loss, _ = generator.pretrain_var_list()
    types = [(0,83-35),(83-35,83)]

    train_batches_songlist = []
    for i in range(Songnum):
        batches, lengthes, features, songnames, lyrics = dataloader.get_batches(BATCH_SIZE, set_no=1,
                                                                feature_dict=dataloader.get_audio_feature(),
                                                                lyric_dict=None, songidx=i,num = num)
        train_batches_songlist.append([batches, lengthes, features, songnames, lyrics])
    valid_batches_songlist = []
    length_list = []
    for i in range(Songnum):
        temp_valid_batches, _, _, _, _ = dataloader.get_batches(BATCH_SIZE, set_no=2,
                                                                feature_dict=dataloader.get_audio_feature(),
                                                                lyric_dict=None, songidx=i)
        valid_batches_songlist.append(temp_valid_batches)
        length_list.append(temp_valid_batches.__len__())

    min_valid_length = 100000
    for i in range(Songnum):
        min_valid_length = min(min_valid_length, valid_batches_songlist[i].__len__())
    for songidx in range(Songnum):
        temp_valid_batches = []
        for batch in valid_batches_songlist[songidx]:
            new_batch = batch.tolist()
            for sen in new_batch:
                temp_valid_batches.append(rm_end(sen, ch2int))
        valid_batches_songlist[songidx] = temp_valid_batches[:BATCH_SIZE * min_valid_length]

    print(min_valid_length * BATCH_SIZE)

    sametype_total_value = 0
    difftype_total_value = 0
    samesong_total_value = 0
    all_total_value = 0
    difftype_max_value = 0
    sametype_max_value = 0
    global_max_value = 0
    for songidx in range(Songnum):
        for type_idx,the_type in enumerate(types):
            if songidx >= the_type[type_idx] and songidx < the_type[type_idx + 1]:
                song_type = type_idx
        if write_json:
            log = open("bleu_log.txt", "a+")
        batches, length, features, songnames, lyrics = train_batches_songlist[songidx]
        if num != None:
            batches = batches[:num]
            lengthes = lengthes[:num]
        for batch_i, batch in enumerate(batches):
            total_length = np.max(lengthes[batch_i])  # 一个batch中的最大长度
            gen_string = []
            feature_batch = features[batch_i]
            feature_batch, feature_length, _ = prepare_feature_batch(feature_batch,songnames[batch_i],dataloader.get_audio_feature())

            feed_dict = {encoder_inputs: feature_batch,
                encoder_lengths: feature_length,
                decoder_lengths: lengthes[batch_i],
                _embed_ph: embedding,
                decoder_inputs: batch}
            if param.Use_lyric:
                lyric_batch = lyrics[batch_i]
                lyric_batch, lyric_length = prepare_lyric_batch(lyric_batch, ch2int)
                feed_dict[encoder_inputs_2] = lyric_batch
                feed_dict[encoder_lengths_2] = lyric_length
            gen_sentence,pretrain_loss_value = sess.run([gen_x,pretrain_loss], feed_dict=feed_dict)
            gen_sentence = gen_sentence.tolist()





            gen_sentence = batch.tolist()




            valid_bleu_list = []
            for i in range(Songnum):
                valid_bleu_list.append(count_bleu(gen_sentence[0:1],ch2int,valid_batches_songlist[i]))


            # char_gen_sentence = []
            # for sen in gen_sentence:
            #     char_gen_sentence.append([])
            #     for ch in sen:
            #         char_gen_sentence[-1].append(int2ch[ch])
            #     print(''.join(char_gen_sentence[-1]))
            valid_bleu_list = np.array(valid_bleu_list)
            sametype_bleu_value = np.mean(valid_bleu_list[types[song_type][0]:types[song_type][1]])
            difftype_bleu_value = np.mean(valid_bleu_list[types[1-song_type][0]:types[1-song_type][1]])
            samesong_bleu_value = valid_bleu_list[songidx]
            all_bleu_value = np.mean(valid_bleu_list)
            sametype_total_value += sametype_bleu_value
            difftype_total_value += difftype_bleu_value
            samesong_total_value += valid_bleu_list[songidx]
            all_total_value += all_bleu_value
            sametype_max_value += np.max(valid_bleu_list[types[song_type][0]:types[song_type][1]])
            difftype_max_value += np.max(
                valid_bleu_list[types[1-song_type][0]:types[1-song_type][1]])
            global_max_value += np.max(valid_bleu_list)
        print('{:>5} {:.5f}({:.5f}) {:.5f}({:.5f}) {:.5f}({:.5f}) {:.5f}({:.5f}) {:.5f}  {:.5f} {:.5f}'.format(
            songidx,sametype_bleu_value,sametype_total_value/(songidx+1)/num,
               difftype_bleu_value,difftype_total_value/(songidx + 1)/num,
              all_bleu_value,all_total_value/(songidx+1)/num,
              samesong_bleu_value,samesong_total_value/(songidx + 1)/num,
            sametype_max_value / (songidx + 1) / num,
            difftype_max_value/ (songidx + 1) / num,
            global_max_value/ (songidx + 1) / num)+time.ctime())
        if write_json:
            log.write('{:>5} {:.5f}({:.5f}) {:.5f}({:.5f}) {:.5f}({:.5f}) {:.5f}({:.5f}) {:.5f}  {:.5f} {:.5f}'.format(
            songidx,sametype_bleu_value,sametype_total_value/(songidx+1)/num,
               difftype_bleu_value,difftype_total_value/(songidx + 1)/num,
              all_bleu_value,all_total_value/(songidx+1)/num,
              samesong_bleu_value,samesong_total_value/(songidx + 1)/num,
            sametype_max_value / (songidx + 1) / num,
            difftype_max_value/ (songidx + 1) / num,
            global_max_value/ (songidx + 1) / num)+time.ctime()+'\n')
            # valid_bleu_value = count_bleu(gen_sentence,ch2int,valid_batches)
            # print(valid_bleu_value)
            log.close()
    return samesong_total_value/Songnum/num,all_total_value/Songnum/num,sametype_max_value/Songnum/num,global_max_value/Songnum/num


def test(num = None, start_word=param.start_length, write_json=False):

    int2ch, ch2int = get_vocab(if_segment)
    param.VOCAB_SIZE = len(int2ch)
    BATCH_SIZE = param.BATCH_SIZE
    _embed_ph = tf.placeholder(tf.float32, [param.VOCAB_SIZE, INPUT_DIM])
    embedding = np.load(os.path.join(data_dir, 'word2vec.npy'))
    learn_rate = tf.Variable(0.0, trainable=False)
    temperature = tf.Variable(1.0,trainable=False)
    end_rate = tf.Variable(1.0,trainable=False)

    dataloader = DataLoader()

    with tf.Session() as sess:
        generator = Generator(encoder_inputs, encoder_lengths,encoder_inputs_2,encoder_lengths_2,decoder_inputs,
                              decoder_lengths  , _embed_ph, learn_rate, ch2int['\\start'],if_test=True,
                              temperature = temperature,end_rate = end_rate)
        generator.init_vars(sess)
        gen_o, gen_x, gen_x_onehot_adv = generator.generate_var_list()
        _, pretrain_loss, _ = generator.pretrain_var_list()
        saver = tf.train.Saver()
        _ = load(saver, sess, param.get_save_dir())





        batches, lengthes, features, songnames, lyrics = dataloader.get_batches(BATCH_SIZE, set_no=4,
                                        feature_dict=dataloader.get_audio_feature(),lyric_dict=None)

        valid_batches, _, _, _, _ = dataloader.get_batches(BATCH_SIZE, set_no=2,
                                        feature_dict=dataloader.get_audio_feature(),lyric_dict=None)
        if num != None:
            batches = batches[:num]
            lengthes = lengthes[:num]
        for batch_i, batch in enumerate(batches):
            total_length = np.max(lengthes[batch_i])  # 一个batch中的最大长度
            gen_string = []
            feature_batch = features[batch_i]
            feature_batch, feature_length,_ = prepare_feature_batch(feature_batch,songnames[batch_i],dataloader.get_audio_feature())

            feed_dict = {encoder_inputs: feature_batch,
                encoder_lengths: feature_length,
                decoder_lengths: lengthes[batch_i],
                _embed_ph: embedding,
                decoder_inputs: batch}
            if param.Use_lyric:
                lyric_batch = lyrics[batch_i]
                lyric_batch, lyric_length = prepare_lyric_batch(lyric_batch, ch2int)
                feed_dict[encoder_inputs_2] = lyric_batch
                feed_dict[encoder_lengths_2] = lyric_length
            gen_sentence,pretrain_loss_value = sess.run([gen_x,pretrain_loss], feed_dict=feed_dict)
            gen_sentence = gen_sentence.tolist()
            batch = batch.tolist()
        _ = load(saver, sess, param.get_adv_save_dir())
        gen_sentence_2, pretrain_loss_value = sess.run([gen_x, pretrain_loss], feed_dict=feed_dict)
        gen_sentence_2 = gen_sentence_2.tolist()

        # valid_bleu_value = count_bleu(gen_sentence,ch2int,valid_batches)
        # print(valid_bleu_value)
        # continue
    for i in range(BATCH_SIZE):  # 为后面的输出batch做处理
        for word_i in range(total_length):
            batch[i][word_i] = int2ch[batch[i][word_i]]
            if batch[i][word_i] == '\end':
                batch[i] = batch[i][0:word_i + 1]
                break
    for sentence in range(BATCH_SIZE):
        print('{:5}'.format(songnames[batch_i][sentence]),end = '')
        for word_i in range(total_length):
            gen_sentence[sentence][word_i] = int2ch[gen_sentence[sentence][word_i]]
            if gen_sentence[sentence][word_i] == '\end' or word_i == (total_length - 1):
                gen_sentence[sentence][-1] = '\end'
                print('生成的句子_MLE: ', end='')
                for word_ii in range(word_i + 1):
                    if word_ii == start_word + 1:
                        print('   ', end='')
                    print(gen_sentence[sentence][word_ii], end='')
                gen_sentence[sentence] = gen_sentence[sentence][0:word_i + 1]
                string = ''.join(gen_sentence[sentence][0:word_i + 1])
                gen_string.append(string[1:-1])
                break
        print('')
        print('{:5}'.format(songnames[batch_i][sentence]),end = '')
        for word_i in range(total_length):
            gen_sentence_2[sentence][word_i] = int2ch[gen_sentence_2[sentence][word_i]]
            if gen_sentence_2[sentence][word_i] == '\end' or word_i == (total_length - 1):
                gen_sentence_2[sentence][-1] = '\end'
                print('生成的句子_GAN: ', end='')
                for word_ii in range(word_i + 1):
                    if word_ii == start_word + 1:
                        print('   ', end='')
                    print(gen_sentence_2[sentence][word_ii], end='')
                gen_sentence_2[sentence] = gen_sentence_2[sentence][0:word_i + 1]
                string = ''.join(gen_sentence_2[sentence][0:word_i + 1])
                gen_string.append(string[1:-1])
                break
        print('')


        print('{:5}'.format(songnames[batch_i][sentence]),end = '')
        print('原来的句子: ', end='')
        for word_i, word in enumerate(batch[sentence]):
            if word_i == start_word + 1:
                print('   ', end='')
            print(batch[sentence][word_i], end='')
        print('')


from discriminator import *

def back_bleu(num = 1,adv = True,batch_size = param.BATCH_SIZE):

    int2ch, ch2int = get_vocab(if_segment)
    param.VOCAB_SIZE = len(int2ch)
    BATCH_SIZE = batch_size
    _embed_ph = tf.placeholder(tf.float32, [param.VOCAB_SIZE, INPUT_DIM])
    learn_rate = tf.Variable(0.0, trainable=False)
    temperature = tf.Variable(1.0,trainable=False)
    end_rate = tf.Variable(1.0,trainable=False)
    embedding = np.load(os.path.join(data_dir, 'word2vec.npy'))
    dataloader = DataLoader()

    sess = tf.Session()
    generator = Generator(encoder_inputs, encoder_lengths, encoder_inputs_2, encoder_lengths_2, decoder_inputs,
                          decoder_lengths , _embed_ph, learn_rate, ch2int['\\start'], if_test=True,
                          temperature = temperature,end_rate = end_rate)
    generator.init_vars(sess)
    gen_o, gen_x, gen_x_onehot_adv = generator.generate_var_list()
    _, pretrain_loss, _ = generator.pretrain_var_list()
    saver = tf.train.Saver()
    if adv is True:
        _ = load(saver, sess, param.get_adv_save_dir())
    else:
        _ = load(saver, sess, param.get_save_dir())
    types = [(0,83-35),(83-35,83)]

    train_batches_songlist = []
    for i in range(Songnum):
        batches, lengthes, features, songnames, lyrics = dataloader.get_batches(BATCH_SIZE, set_no=4,
                                                                feature_dict=dataloader.get_audio_feature(),
                                                                lyric_dict=None, songidx=i)
        train_batches_songlist.append([batches, lengthes, features, songnames, lyrics])
    valid_batches_songlist = []
    length_list = []
    for i in range(Songnum):
        temp_valid_batches, _, _, _, _ = dataloader.get_batches(BATCH_SIZE, set_no=2,
                                                                feature_dict=dataloader.get_audio_feature(),
                                                                lyric_dict=None, songidx=i)
        valid_batches_songlist.append(temp_valid_batches)
        length_list.append(temp_valid_batches.__len__())

    min_valid_length = 100000
    for i in range(Songnum):
        min_valid_length = min(min_valid_length, valid_batches_songlist[i].__len__())
    for songidx in range(Songnum):
        temp_valid_batches = []
        for batch in valid_batches_songlist[songidx]:
            new_batch = batch.tolist()
            for sen in new_batch:
                temp_valid_batches.append(rm_end(sen, ch2int))
        valid_batches_songlist[songidx] = temp_valid_batches[:BATCH_SIZE * min_valid_length]

    print(min_valid_length * BATCH_SIZE)

    for songidx in range(Songnum):
        batches, lengthes, features, songnames, lyrics = train_batches_songlist[songidx]
        gen_sentences = []
        for batch_i in range(min_valid_length):
            batch = batches[batch_i]
            feature_batch = features[batch_i]
            feature_batch, feature_length, _ = prepare_feature_batch(feature_batch,songnames[batch_i],dataloader.get_audio_feature())

            feed_dict = {encoder_inputs: feature_batch,
                encoder_lengths: feature_length,
                decoder_lengths: lengthes[batch_i],
                _embed_ph: embedding,
                decoder_inputs: batch}
            if param.Use_lyric:
                lyric_batch = lyrics[batch_i]
                lyric_batch, lyric_length = prepare_lyric_batch(lyric_batch, ch2int)
                feed_dict[encoder_inputs_2] = lyric_batch
                feed_dict[encoder_lengths_2] = lyric_length
            gen_sentence = sess.run(gen_x, feed_dict=feed_dict)
            gen_sentences.extend(gen_sentence.tolist())
        train_gen_sentence = []
        for sen in gen_sentences:
            train_gen_sentence.append(rm_end(sen,ch2int))
        train_batches_songlist[songidx] = train_gen_sentence
        print(songidx)


    sametype_total_value = 0
    difftype_total_value = 0
    samesong_total_value = 0
    all_total_value = 0
    difftype_max_value = 0
    sametype_max_value = 0
    global_max_value = 0
    for songidx in range(Songnum):
        for type_idx,the_type in enumerate(types):
            if songidx >= the_type[type_idx] and songidx < the_type[type_idx + 1]:
                song_type = type_idx
        train_bleu_list = []
        for i in range(num):
            for song_idx in range(Songnum):
                train_bleu_list.append(count_bleu(valid_batches_songlist[songidx][i:i+1],
                                              ch2int, train_batches_songlist[song_idx]))
            train_bleu_list = np.array(train_bleu_list)

            sametype_bleu_value = np.mean(train_bleu_list[types[song_type][0]:types[song_type][1]])
            difftype_bleu_value = np.mean(train_bleu_list[types[1 - song_type][0]:types[1 - song_type][1]])
            samesong_bleu_value = train_bleu_list[songidx]
            all_bleu_value = np.mean(train_bleu_list)
            sametype_total_value += sametype_bleu_value
            difftype_total_value += difftype_bleu_value
            samesong_total_value += train_bleu_list[songidx]
            all_total_value += all_bleu_value
            sametype_max_value += np.max(train_bleu_list[types[song_type][0]:types[song_type][1]])
            difftype_max_value += np.max(
                train_bleu_list[types[1 - song_type][0]:types[1 - song_type][1]])
            global_max_value += np.max(train_bleu_list)

        print('{:>5} {:.5f}({:.5f}) {:.5f}({:.5f}) {:.5f}({:.5f}) {:.5f}({:.5f}) {:.5f}  {:.5f} {:.5f}'.format(
            songidx,sametype_bleu_value,sametype_total_value/(songidx+1)/num,
               difftype_bleu_value,difftype_total_value/(songidx + 1)/num,
              all_bleu_value,all_total_value/(songidx+1)/num,
              samesong_bleu_value,samesong_total_value/(songidx + 1)/num,
            sametype_max_value / (songidx + 1) / num,
            difftype_max_value/ (songidx + 1) / num,
            global_max_value/ (songidx + 1) / num)+time.ctime())
    return samesong_total_value/Songnum/num,all_total_value/Songnum/num,\
           sametype_max_value/Songnum/num,global_max_value/Songnum/num



def dis_test(adv = True,test = True):
    int2ch, ch2int = get_vocab(if_segment)
    param.VOCAB_SIZE = len(int2ch)
    BATCH_SIZE = param.BATCH_SIZE
    _embed_ph = tf.placeholder(tf.float32, [param.VOCAB_SIZE, INPUT_DIM])
    learn_rate = tf.Variable(0.0, trainable=False)
    temperature = tf.Variable(1.0,trainable=False)
    end_rate = tf.Variable(1.0,trainable=False)
    embedding = np.load(os.path.join(data_dir, 'word2vec.npy'))
    dataloader = DataLoader()
    max_len = tf.shape(decoder_inputs)[-1]
    log = open("E_val.txt", "a+")

    with tf.Session() as sess:
        generator = Generator(encoder_inputs, encoder_lengths,encoder_inputs_2,encoder_lengths_2,decoder_inputs,
                              decoder_lengths  , _embed_ph, learn_rate, ch2int['\\start'],if_test=True,
                              temperature=temperature,end_rate=end_rate)
        generator.init_vars(sess)
        gen_o, gen_x, gen_x_onehot_adv = generator.generate_var_list()
        _, pretrain_loss, original_loss = generator.pretrain_var_list()
        encoder_final_state = generator.encoder_state_var()

        saver = tf.train.Saver()
        if adv:
            _ = load(saver, sess, param.get_adv_save_dir())
        else:
            _ = load(saver, sess, param.get_save_dir())


        discriminator_func = tf.make_template('discriminator_3', discriminator_2)
        song_label_2 = tf.reshape(tf.one_hot(song_label, Songnum), [BATCH_SIZE, 1, Songnum])
        real_onehot = tf.one_hot(decoder_inputs[:,0:max_len], param.VOCAB_SIZE, 1.0, 0.0)

        umatch_feature = tf.concat([tf.concat([song_label_2[1:], song_label_2[0:1]], 0),
                                    tf.concat([encoder_final_state[1:], encoder_final_state[0:1]], 0)], -1)
        match_feature = tf.concat([song_label_2, encoder_final_state], -1)
        d_out_match = discriminator_func(real_onehot, match_feature, BATCH_SIZE, max_len, VOCAB_SIZE, embed_ph)
        d_out_umatch = discriminator_func(real_onehot, umatch_feature, BATCH_SIZE, max_len, VOCAB_SIZE, embed_ph)
        d_out_match_fake = discriminator_func(tf.nn.softmax(tf.multiply(gen_x_onehot_adv,1000)), match_feature, BATCH_SIZE, max_len, VOCAB_SIZE,
                                                embed_ph)
        d_out_umatch_fake = discriminator_func(tf.nn.softmax(tf.multiply(gen_x_onehot_adv,1000)), umatch_feature, BATCH_SIZE, max_len, VOCAB_SIZE,
                                                 embed_ph)
        _, d_loss_2_1 = get_losses(d_out_match, d_out_umatch, gan_type=param.gan_type)
        g_loss_2, d_loss_2_3 = get_losses(d_out_match, d_out_match_fake, gan_type=param.gan_type)
        _, d_loss_2_2 = get_losses(d_out_match, d_out_umatch_fake, gan_type=param.gan_type)
        _, d_loss_2_4 = get_losses(d_out_match_fake, d_out_umatch_fake, gan_type=param.gan_type)

        d_loss_2 = d_loss_2_1
        d_lr = learn_rate
        grad_clip = 5
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_3')
        d_optimizer = tf.train.AdamOptimizer(d_lr, beta1=0.9, beta2=0.999)
        d_grads_1, _ = tf.clip_by_global_norm(tf.gradients(d_loss_2, d_vars), grad_clip)  # gradient clipping
        d_train_op = d_optimizer.apply_gradients(zip(d_grads_1, d_vars))
        d_1 = tf.reduce_mean(tf.nn.sigmoid(d_out_match))
        d_2 = tf.reduce_mean(tf.nn.sigmoid(d_out_umatch))
        d_3 = tf.reduce_mean(tf.nn.sigmoid(d_out_match_fake))
        #d_3 = tf.reduce_mean(tf.to_float(tf.greater(d_out_match_fake,0.5)))
        d_4 = tf.reduce_mean(tf.nn.sigmoid(d_out_umatch_fake))
        saver_3 = tf.train.Saver(var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_3'))
        batches, lengthes, features, songnames, lyrics = dataloader.get_batches(BATCH_SIZE, set_no=2,
                                        feature_dict=dataloader.get_audio_feature(),lyric_dict=None)
        last_batch_i = load(saver_3, sess, './data/model/EGAN')
        if (last_batch_i is not None) and (last_batch_i > 9000):
            learn_rate.assign(sess.run(learn_rate) * 0)
            print('Not training')
        total_d_loss = 0
        range_d_loss = 200
        uninitialized_vars = find_uninitialized_var(sess)
        initialize_op = tf.variables_initializer(uninitialized_vars)
        sess.run(initialize_op)
        for epoch_i in range(1):
            batches = batches[0:10000]
            total_d_1,total_d_2,total_d_3,total_d_4 = 0,0,0,0
            for batch_i in range(10000):
                if test:
                    batches, lengthes, features, songnames, lyrics = dataloader.get_batches(BATCH_SIZE, set_no=2,
                                                                                        feature_dict=dataloader.get_audio_feature(),
                                                                                        lyric_dict=None,songidx=batch_i,num=1)
                    batch = batches[0]
                else:
                    batch = batches[batch_i]
                total_length = np.max(lengthes[batch_i])  # 一个batch中的最大长度
                gen_string = []
                feature_batch = features[batch_i]
                feature_batch, feature_length,song_label_value = prepare_feature_batch(feature_batch,songnames[batch_i],dataloader.get_audio_feature())

                feed_dict = {encoder_inputs: feature_batch,
                    encoder_lengths: feature_length,
                    decoder_lengths: lengthes[batch_i],
                    _embed_ph: embedding,
                    decoder_inputs: batch,
                    song_label:song_label_value}
                if param.Use_lyric:
                    lyric_batch = lyrics[batch_i]
                    lyric_batch, lyric_length = prepare_lyric_batch(lyric_batch, ch2int)
                    feed_dict[encoder_inputs_2] = lyric_batch
                    feed_dict[encoder_lengths_2] = lyric_length
                _,d_loss_value,d_1_val,d_2_val,d_3_val,d_4_val = sess.run([d_train_op,d_loss_2_4,d_1,d_2,d_3,d_4],
                                                                          feed_dict=feed_dict)
                total_d_loss += d_loss_value
                total_d_1 += d_1_val
                total_d_2 += d_2_val
                total_d_3 += d_3_val
                total_d_4 += d_4_val
                range_d_loss = range_d_loss * 0.99 + d_loss_value
                if test and batch_i==Songnum-1:
                    break
                if batch_i % 100 ==99:
                    bi = batch_i % 100 + 1
                    learn_rate.assign(sess.run(learn_rate)*0.95)
                    string = '{:>3} {:>3}/{:>3} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
                        epoch_i,batch_i+1,len(batches),d_loss_value,range_d_loss/100,total_d_loss/(batch_i + 1),
                    total_d_1/bi,total_d_2/bi,total_d_3/bi,total_d_4/bi)
                    print(string)
                    log.write(string + '\n')
                    total_d_1,total_d_2,total_d_3,total_d_4 = 0,0,0,0
                    #save(saver_3, sess, './data/model/EGAN', batch_i//100)
            bi = batch_i % 100 + 1
            learn_rate.assign(sess.run(learn_rate) * 0.95)
            string = '{:>3} {:>3}/{:>3} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
                epoch_i, batch_i + 1, len(batches), d_loss_value, range_d_loss / 100,
                         total_d_loss / (batch_i + 1),
                         total_d_1 / bi, total_d_2 / bi, total_d_3 / bi, total_d_4 / bi)
            print(string)
            log.write(string + '\n')
            total_d_1, total_d_2, total_d_3, total_d_4 = 0, 0, 0, 0
                    # if last_batch_i>9000:
                    #     break


def count_pro():
    int2ch, ch2int = get_vocab(if_segment)
    param.VOCAB_SIZE = len(int2ch)
    BATCH_SIZE = param.BATCH_SIZE
    _embed_ph = tf.placeholder(tf.float32, [param.VOCAB_SIZE, INPUT_DIM])
    learn_rate = tf.Variable(0.0, trainable=False)
    temperature = tf.Variable(1.0,trainable=False)
    end_rate = tf.Variable(1.0,trainable=False)
    embedding = np.load(os.path.join(data_dir, 'word2vec.npy'))
    dataloader = DataLoader()
    with tf.Session() as sess:
        generator = Generator(encoder_inputs, encoder_lengths,encoder_inputs_2,encoder_lengths_2,decoder_inputs,
                              decoder_lengths  , _embed_ph, learn_rate, ch2int['\\start'],if_test=True,
                              temperature=temperature,end_rate=end_rate)
        generator.init_vars(sess)
        gen_o, gen_x, gen_x_onehot_adv = generator.generate_var_list()
        _, pretrain_loss, _ = generator.pretrain_var_list()
        logits = generator.pretrain_logits()
        saver = tf.train.Saver()
        _ = load(saver, sess, param.get_save_dir())
        batches, lengthes, features, songnames, lyrics = dataloader.get_batches(BATCH_SIZE, set_no=2,
                                        feature_dict=dataloader.get_audio_feature(),lyric_dict=None)
        total_pro_value = 0

        for batch_i, batch in enumerate(batches):
            total_length = np.max(lengthes[batch_i])  # 一个batch中的最大长度
            gen_string = []
            feature_batch = features[batch_i]
            feature_batch, feature_length,_ = prepare_feature_batch(feature_batch,songnames[batch_i],dataloader.get_audio_feature())

            feed_dict = {encoder_inputs: feature_batch,
                encoder_lengths: feature_length,
                decoder_lengths: lengthes[batch_i],
                _embed_ph: embedding,
                decoder_inputs: batch}
            if param.Use_lyric:
                lyric_batch = lyrics[batch_i]
                lyric_batch, lyric_length = prepare_lyric_batch(lyric_batch, ch2int)
                feed_dict[encoder_inputs_2] = lyric_batch
                feed_dict[encoder_lengths_2] = lyric_length
            logits_value,loss_value = sess.run([tf.nn.softmax(logits,2),pretrain_loss], feed_dict=feed_dict)
            for i in range(BATCH_SIZE):
                total_pro_value += 1/scipy.stats.gmean((logits_value[i].T)[batch[i, 1:lengthes[batch_i][i]]].diagonal() + 1e-6)/BATCH_SIZE
            if batch_i%10 == 9:
                print(batch_i,len(batches),total_pro_value/(batch_i + 1))
        return total_pro_value/(len(batches))

def count_vocab(adv = True):
    int2ch, ch2int = get_vocab(if_segment)
    param.VOCAB_SIZE = len(int2ch)
    BATCH_SIZE = param.BATCH_SIZE
    _embed_ph = tf.placeholder(tf.float32, [param.VOCAB_SIZE, INPUT_DIM])
    learn_rate = tf.Variable(0.0, trainable=False)
    temperature = tf.Variable(1.0,trainable=False)
    end_rate = tf.Variable(1.0,trainable=False)
    embedding = np.load(os.path.join(data_dir, 'word2vec.npy'))
    dataloader = DataLoader()
    with tf.Session() as sess:
        generator = Generator(encoder_inputs, encoder_lengths,encoder_inputs_2,encoder_lengths_2,decoder_inputs,
                              decoder_lengths  , _embed_ph, learn_rate, ch2int['\\start'],if_test=True,
                              temperature=temperature,end_rate=end_rate)
        generator.init_vars(sess)
        gen_o, gen_x, gen_x_onehot_adv = generator.generate_var_list()
        _, pretrain_loss, _ = generator.pretrain_var_list()
        logits = generator.pretrain_logits()
        saver = tf.train.Saver()
        batches, lengthes, features, songnames, lyrics = dataloader.get_batches(BATCH_SIZE, set_no=2,
                                        feature_dict=dataloader.get_audio_feature(),lyric_dict=None,num=10)

        if adv:
            _ = load(saver, sess, param.get_adv_save_dir())
        else:
            _ = load(saver, sess, param.get_save_dir())


        total_pro_value = 0
        total_onehot = np.zeros([param.VOCAB_SIZE])
        for batch_i, batch in enumerate(batches[0:10]):
            total_length = np.max(lengthes[batch_i])  # 一个batch中的最大长度
            gen_string = []
            feature_batch = features[batch_i]
            feature_batch, feature_length,_ = prepare_feature_batch(feature_batch,songnames[batch_i],dataloader.get_audio_feature())

            feed_dict = {encoder_inputs: feature_batch,
                encoder_lengths: feature_length,
                decoder_lengths: lengthes[batch_i],
                _embed_ph: embedding,
                decoder_inputs: batch}
            if param.Use_lyric:
                lyric_batch = lyrics[batch_i]
                lyric_batch, lyric_length = prepare_lyric_batch(lyric_batch, ch2int)
                feed_dict[encoder_inputs_2] = lyric_batch
                feed_dict[encoder_lengths_2] = lyric_length
            gen_sentence = sess.run(tf.one_hot(gen_x,param.VOCAB_SIZE,1.0,0.0), feed_dict=feed_dict)
            gen_sentence = np.sum(gen_sentence, 0)
            gen_sentence = np.sum(gen_sentence, 0)
            total_onehot = total_onehot+gen_sentence
        print(np.sum(np.float32(total_onehot>0)))
        return np.sum(np.float32(total_onehot>0))




from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction



def count_bleu(gen_sentence,ch2int,valid_batches):
    def valid_bleu(sentence, valid_batches):
        # reference = [[1,5,2,5,2], [12,2,5,2,3,5,1,3,5,2]]
        # candidate = [1,2,3,4,12,31,2,5,2,5,2]
        score = sentence_bleu(valid_batches, sentence, weights=param.bleu_weights,smoothing_function=SmoothingFunction().method1)
        return score
    bleu_value = 0
    for i in range(len(gen_sentence)):
        sen = rm_end(gen_sentence[i], ch2int)
        sens = []
        while len(sen) > 10:
            sens.append(sen[:10])
            sen = sen[10:]
        for sen in sens:
            bleu_value += valid_bleu(sen, valid_batches)/len(sens)
    bleu_value /= len(gen_sentence)
    return bleu_value
def output_process(gen_sentence,int2ch):
    total_length = 30
    for word_i in range(total_length):
        gen_sentence[word_i] = int2ch[gen_sentence[word_i]]
        if gen_sentence[word_i] == '\end' or word_i == (total_length - 1):
            gen_sentence[-1] = '\end'
            print('生成的句子_MLE: ', end='')
            for word_ii in range(word_i + 1):
                print(gen_sentence[word_ii], end='')
            gen_sentence = gen_sentence[0:word_i + 1]
            string = ''.join(gen_sentence[0:word_i + 1])
            break
    print('')

if __name__ == '__main__':
#    test(1,start_word=param.start_length,write_json= False)
#    compare_bleu(4,write_json = True,model='mle')
#    count_pro()
    dis_test(True)
