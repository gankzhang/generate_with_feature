from generate import  *
import warnings
warnings.filterwarnings("ignore")
from param import *
from discriminator import  discriminator
from discriminator_2 import discriminator_2
import tensorflow.contrib.slim as slim
from data_utils import *
from test import rm_end,count_bleu,count_pro,compare_bleu,output_process
import scipy
def pretrain(n_epochs = 100):
    int2ch, ch2int = get_vocab(if_segment())
    param.VOCAB_SIZE = len(int2ch)
    BATCH_SIZE = param.BATCH_SIZE

    learn_rate = tf.Variable(0.0, trainable=False)
    temperature = tf.Variable(1.0,trainable=False)
    end_rate = tf.Variable(1.0,trainable=False)
    last_valid_loss = 0
    last_train_loss = 0
    max_len = tf.shape(decoder_inputs)[-1]
    dataloader = DataLoader()
    print("Start training RNN enc-dec model ...")
    embedding = np.load(os.path.join(data_dir ,'word2vec.npy'))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        generator = Generator(encoder_inputs,encoder_lengths,encoder_inputs_2,encoder_lengths_2,decoder_inputs,
                    decoder_lengths,embed_ph,learn_rate,ch2int['\\start'],False,temperature,end_rate)
        if Use_Weight:
            decoder_final_state, loss, original_loss = generator.pretrain_var_list(mean_loss=False)
        else:
            decoder_final_state, loss, original_loss = generator.pretrain_var_list()

        if Use_VAE:
            mn,sd = generator.VAE_variable()
            latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
            loss = loss + latent_loss
        if Use_Weight:
            encoder_final_state = generator.encoder_state_var()
            discriminator_func_2 = tf.make_template('discriminator_2', discriminator)
            song_label_2 = tf.reshape(tf.one_hot(song_label, Songnum), [BATCH_SIZE, 1, Songnum])
            real_onehot = tf.one_hot(decoder_inputs[:, 0:max_len], param.VOCAB_SIZE, 1.0, 0.0)
            umatch_feature = tf.concat([tf.concat([song_label_2[1:],song_label_2[0:1]],0),
                                        tf.concat([encoder_final_state[1:],encoder_final_state[0:1]],0)],-1)
            match_feature = tf.concat([song_label_2,encoder_final_state],-1)
            d_out_match = discriminator_func_2(real_onehot,match_feature , BATCH_SIZE, max_len, VOCAB_SIZE, embed_ph)
            d_out_umatch = discriminator_func_2(real_onehot,umatch_feature,BATCH_SIZE,max_len,VOCAB_SIZE,embed_ph)
            _, d_loss_2 = get_losses(d_out_match, d_out_umatch, gan_type=param.gan_type)
            _, _, d_train_op_2 = get_train_ops(None, None, d_loss_2, temperature, 0, learn_rate)
            prob_match = tf.reduce_mean(tf.reshape(tf.sigmoid(d_out_match), [BATCH_SIZE, 64]), 1)
            prob_umatch = tf.reduce_mean(tf.reshape(tf.sigmoid(d_out_match), [BATCH_SIZE, 64]), 1)
            mean, variance = tf.nn.moments(prob_match,axes=[0])
            weight = tf.nn.softmax(tf.truediv(prob_match-mean,2*(variance+1e-7)))
            loss = tf.reduce_mean(tf.multiply(weight,loss))
            train_op = get_pretrain_op(loss,learn_rate)

        else:
            loss = tf.reduce_mean(loss)
            train_op = get_pretrain_op(loss,learn_rate)
        _, gen_x, _ = generator.generate_var_list()
        logits = tf.nn.softmax(generator.pretrain_logits(),2)
        encoder_final_state =generator.encoder_state_var()
        right_word,gen_length = generator.right_word_count()
        reco_op,reco_loss = get_reco_op(encoder_final_state,song_label,learn_rate)



        generator.init_vars(sess)

        saver = tf.train.Saver()
        saved_global_step = load(saver, sess, save_dir)
        if saved_global_step is None:
            saved_global_step = -1
        feature_dict = dataloader.get_audio_feature()#feature should be
        lyric_dict = dataloader.get_lyric()
        for epoch in range(saved_global_step + 1 ,n_epochs):
            log = open("log.txt", "a+")
            batches, length, features, songnames, lyrics = dataloader.get_batches(batch_size=BATCH_SIZE, set_no = 1
                                                                  ,feature_dict=feature_dict,lyric_dict =None)
            valid_batches, _, _, valid_songnames, _ = dataloader.get_batches(BATCH_SIZE, set_no=2,
                                                               feature_dict=feature_dict,
                                                               lyric_dict=None)
            valid_batches = process_for_bleu(valid_batches,ch2int)

            sess.run(learn_rate.assign(param.LEARNRATE * decay_rate ** (epoch // 5 * 5)))
            total_loss = 0
            total_original_loss = 0
            logits_total_prob = 0
            total_reco_loss = 0
            right_word_total_value = 0
            batches = batches
            gen_total_length = 0
            for batch_i ,batch in enumerate(batches):
                feature_batch = features[batch_i]
                feature_batch, feature_length,song_label_value = \
                    prepare_feature_batch(feature_batch,songnames[batch_i], feature_dict)

                feed_dict = {encoder_inputs: feature_batch,
                             encoder_lengths: feature_length,
                             decoder_lengths: length[batch_i],
                             embed_ph: embedding,
                             decoder_inputs: batch,
                             song_label:song_label_value}
                if param.Use_lyric:
                    lyric_batch = lyrics[batch_i]
                    lyric_batch, lyric_length = prepare_lyric_batch(lyric_batch, ch2int)
                    feed_dict[encoder_inputs_2] = lyric_batch
                    feed_dict[encoder_lengths_2] = lyric_length

                loss_value,reco_loss_value ,_,_,original_loss_value,logits_value,right_word_value,gen_length_value = \
                    sess.run([loss ,reco_loss, train_op,reco_op, original_loss, logits, right_word,gen_length], feed_dict=feed_dict)
                total_loss += loss_value
                total_original_loss += original_loss_value
                total_reco_loss += reco_loss_value
                right_word_total_value += right_word_value
                gen_total_length += gen_length_value
                for sen_id in range(BATCH_SIZE):
                    logits_total_prob += 1/scipy.stats.gmean((logits_value[sen_id].T)[batch[sen_id, 1:length[batch_i][sen_id]]].diagonal()
                                                             + 1e-6)/BATCH_SIZE
                if batch_i  %  10 == 9:
                    print('Epoch {:>3} Batch {:>4}/{}  reco_loss = {:.3f} train_loss = {:.3f} avg_loss = {:.3f}'
                          ' avg_prob = {:.3f} right = {:.3f} gen_len = {:.3f} end_rate = {:.3f}'.format(
                     epoch ,batch_i + 1 ,len(batches),reco_loss_value ,loss_value ,total_loss /(batch_i + 1),logits_total_prob/(batch_i + 1),
                    right_word_total_value/(batch_i + 1),gen_length_value,sess.run(end_rate)))
                    gen_sentence = sess.run([gen_x],feed_dict=feed_dict)
                    gen_sentence = gen_sentence[0].tolist()
                    output_process(gen_sentence[0],int2ch)
                sess.run(tf.assign_add(end_rate,((gen_length_value/expected_length - 1))/100))


#            avg_valid_bleu = count_bleu(gen_sentence[:valid_batch_num],ch2int,valid_batches)
            avg_valid_bleu = 0#compare_bleu(num=1, sess=sess, write_json=False, generator=generator, batch_size=param.BATCH_SIZE)
            avg_valid_prob,avg_valid_loss = valid(sess, logits, original_loss, embedding)
            bn = len(batches)
            string = 'Epoch {:>3}   total_loss = {:.3f}({:.3f}),avg = {:.3f},avg_ori = {:.3f} ,valid_avg = {:.3f},' \
                     'valid_prob = {:.3f} bleu ={:.3f}'\
                     'reco_loss = {:.3f} right = {:.4f} lr = {:.5f} len={:.3f}'.format(
                     epoch ,total_loss ,total_loss - last_train_loss ,total_loss / bn,total_original_loss/bn
                     ,avg_valid_loss,avg_valid_prob,avg_valid_bleu,total_reco_loss/bn,right_word_total_value/bn
                     ,sess.run(learn_rate),gen_total_length/bn) + time.ctime()
            print(string)
            log.write(string + '\n')
            last_train_loss = total_loss
            # save
            if epoch % 1 == 0:
                save(saver, sess, save_dir, epoch)
            print('Model Trained and Saved')
            log.close()


def adv_train(n_epochs = 100):
    int2ch, ch2int = get_vocab(if_segment())
    param.VOCAB_SIZE = len(int2ch)
    BATCH_SIZE = param.BATCH_SIZE

    learn_rate = tf.Variable(0.0, trainable=False)
    temperature = tf.Variable(1.0,trainable=False)
    end_rate = tf.Variable(1.0,trainable=False)
    max_len = tf.shape(decoder_inputs)[-1]
        # max_len = tf.reduce_max(decoder_lengths)

    dataloader = DataLoader()
    print("Start training RNN enc-dec model ...")
    embedding = np.load(os.path.join(data_dir, 'word2vec.npy'))
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        generator = Generator(encoder_inputs, encoder_lengths,encoder_inputs_2, encoder_lengths_2, decoder_inputs,
                              decoder_lengths, embed_ph, learn_rate, ch2int['\\start'],False,
                              temperature = temperature,end_rate=end_rate)
        _, pretrain_loss, original_loss = generator.pretrain_var_list()
        pre_train_op = get_pretrain_op(pretrain_loss, learn_rate * pre_lr_rate)
        gen_o, gen_x, gen_x_onehot_adv = generator.generate_var_list()
        encoder_final_state = generator.encoder_state_var()
        reco_op, reco_loss = get_reco_op(encoder_final_state, song_label, learn_rate/100)
        #prob, tokens, gumbeled prob

        generator.init_vars(sess)
        saved_global_step = load(tf.train.Saver(), sess, save_dir)
        if saved_global_step is None:
            saved_global_step = -1
            generator.init_vars(sess)


        if Use_VAE:
            decoder_inputs_onehot = tf.one_hot(decoder_inputs[:,:max_len],param.VOCAB_SIZE)
            rege_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=gen_x_onehot_adv,
                labels=decoder_inputs_onehot)
            mn, sd = generator.VAE_variable()
            latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
            vae_op = get_pretrain_op(tf.reduce_mean(rege_loss) + tf.reduce_mean(latent_loss), learn_rate/10)


        feature_dict = dataloader.get_audio_feature()  # feature should be
        discriminator_func = tf.make_template('discriminator', discriminator)
        discriminator_func_2 = tf.make_template('discriminator_2', discriminator_2)
        d_out_fake = discriminator_func(tf.nn.softmax(gen_x_onehot_adv),encoder_final_state,BATCH_SIZE,max_len,VOCAB_SIZE,embed_ph,False)
        real_onehot = tf.one_hot(decoder_inputs[:,0:max_len], param.VOCAB_SIZE, 1.0, 0.0)
        d_out_real = discriminator_func(real_onehot,encoder_final_state, BATCH_SIZE, max_len, VOCAB_SIZE, embed_ph,False)
        if D_with_state:
            song_label_2 = tf.reshape(tf.one_hot(song_label, Songnum), [BATCH_SIZE, 1, Songnum])
            umatch_feature = tf.concat([tf.concat([song_label_2[1:],song_label_2[0:1]],0),
                                        tf.concat([encoder_final_state[1:],encoder_final_state[0:1]],0)],-1)
            match_feature = tf.concat([song_label_2,encoder_final_state],-1)
            d_out_match,d_state_match = discriminator_func_2(real_onehot,match_feature , BATCH_SIZE, max_len, VOCAB_SIZE, embed_ph)
            d_out_umatch,d_state_umatch = discriminator_func_2(real_onehot,umatch_feature,BATCH_SIZE,max_len,VOCAB_SIZE,embed_ph)

            gen_x_onehot_adv_2 = tf.nn.softmax(tf.multiply(gen_x_onehot_adv,1))
            d_out_match_fake,d_state_match_fake = discriminator_func_2(gen_x_onehot_adv_2,match_feature,BATCH_SIZE, max_len, VOCAB_SIZE, embed_ph)
            d_out_umatch_fake,d_state_umatch_fake = discriminator_func_2(gen_x_onehot_adv_2,umatch_feature,BATCH_SIZE,max_len,VOCAB_SIZE,embed_ph)





            dor = tf.reduce_mean(tf.reshape(tf.sigmoid(d_out_real), [BATCH_SIZE, 64]), 1)[0]
            dof = tf.reduce_mean(tf.reshape(tf.sigmoid(d_out_fake), [BATCH_SIZE, 64]), 1)[0]
            dom = tf.reduce_mean(tf.reshape(tf.sigmoid(d_out_match),[BATCH_SIZE,64]),1)[0]
            dou = tf.reduce_mean(tf.reshape(tf.sigmoid(d_out_umatch), [BATCH_SIZE, 64]), 1)[0]
            domf = tf.reduce_mean(tf.reshape(tf.sigmoid(d_out_match_fake), [BATCH_SIZE, 64]), 1)[0]
            douf = tf.reduce_mean(tf.reshape(tf.sigmoid(d_out_umatch_fake), [BATCH_SIZE, 64]), 1)[0]





            # g_loss, d_loss = get_losses(d_out_real, tf.concat([d_out_fake,d_out_umatch],0), gan_type=param.gan_type)
            g_loss_1, d_loss_1 = get_losses(d_out_real, d_out_fake, gan_type=param.gan_type)
            _, d_loss_2_1 = get_losses(d_out_match, d_out_umatch, gan_type=param.gan_type)
            g_loss_2, d_loss_2_3 = get_losses(d_out_match, d_out_match_fake, gan_type=param.gan_type)
            _, d_loss_2_2 = get_losses(d_out_match, d_out_umatch_fake, gan_type=param.gan_type)
            g_loss_3 = tf.reduce_mean(tf.square(tf.subtract(d_state_match, d_state_match_fake)))
            g_loss = (g_loss_1 +  g_loss_2 + g_loss_3)/3
            d_loss_2 = (d_loss_2_1)/1
        else:
            g_loss, d_loss = get_losses(d_out_real, d_out_fake, gan_type=param.gan_type)
        g_train_op, d_train_op_1,d_train_op_2 = get_train_ops(g_loss, d_loss_1,d_loss_2, temperature, 0,learn_rate)#change the global step to a Variable
        logits = tf.nn.softmax(generator.pretrain_logits(),2)
        right_word,gen_length = generator.right_word_count()
        adv_saved_global_step = load(tf.train.Saver(), sess, get_adv_save_dir())
        if adv_saved_global_step is not None:
            saved_global_step = adv_saved_global_step
        else:
            variables = slim.get_variables(scope="discriminator")
            init_op = tf.variables_initializer(variables)
            sess.run(init_op)

        uninitialized_vars = find_uninitialized_var(sess)
        for var in uninitialized_vars:
            print(var)
        initialize_op = tf.variables_initializer(uninitialized_vars)
        sess.run(initialize_op)

        for epoch in range(saved_global_step + 1, n_epochs):
            uninitialized_vars = find_uninitialized_var(sess)
            initialize_op = tf.variables_initializer(uninitialized_vars)
            sess.run(initialize_op)


            log = open("log.txt", "a+")
            print('start prepocessing')
            batches, length, features, songnames, lyrics = dataloader.get_batches(batch_size=BATCH_SIZE, set_no = 1,
                                                                          feature_dict=feature_dict,lyric_dict=None,
                                                                          num=adv_train_batch_num)
            valid_batches, _, _, _, _ = dataloader.get_batches(BATCH_SIZE, set_no=2,
                                                               feature_dict=dataloader.get_audio_feature(), lyric_dict=None,
                                                               num=valid_batch_num)
            valid_batches = process_for_bleu(valid_batches, ch2int)
            print('end prepocessing')

            sess.run(learn_rate.assign(param.LEARNRATE * decay_rate ** (epoch // 5 * 5)))
            d_total_loss = 0
            d_total_loss_1 = 0
            d_total_loss_2 = 0
            g_total_loss_1 = 0
            g_total_loss_2 = 0
            pre_total_loss = 0
            logits_total_prob = 0
            right_word_total_value = 0
            total_gen_length = 0
            ac_val_1,ac_val_2,ac_val_3,ac_val_4,ac_val_5,ac_val_6 = 0,0,0,0,0,0
            batches = batches[0:adv_train_batch_num]
            for batch_i, batch in enumerate(batches):
                sess.run(temperature.assign(min((param.temperature **
                                            (((epoch - adv_start_step) * adv_train_batch_num + batch_i) / (
                                                        n_adv_step * adv_train_batch_num))),param.temperature)))

                feature_batch = features[batch_i]
                feature_batch, feature_length, song_label_value = \
                    prepare_feature_batch(feature_batch, songnames[batch_i], feature_dict)
                feed_dict = {encoder_inputs: feature_batch,
                             encoder_lengths: feature_length,
                             decoder_lengths: length[batch_i],
                             embed_ph: embedding,
                             decoder_inputs: batch,
                             song_label:song_label_value}
                if param.Use_lyric:
                    lyric_batch = lyrics[batch_i]
                    lyric_batch, lyric_length = prepare_lyric_batch(lyric_batch, ch2int)
                    feed_dict[encoder_inputs_2] = lyric_batch
                    feed_dict[encoder_lengths_2] = lyric_length
                place = [reco_op, g_train_op, d_train_op_1,d_train_op_2, g_loss_1, g_loss_2,g_loss_3,d_loss_1,d_loss_2, pretrain_loss,reco_loss, gen_x, logits,right_word,gen_length,
                     dor,dof,dom,dou,domf,douf]
                if Use_VAE:
                    place.extend([vae_op,rege_loss,latent_loss])

                result = sess.run(place,feed_dict=feed_dict)
                _, _, _,_, g_loss_value_1, g_loss_value_2,g_loss_value_3, d_loss_value_1, d_loss_value_2, \
                pretrain_loss_value, reco_loss_value, gen_sentence, logits_value, right_word_value,gen_length_value, \
                ac1, ac2, ac3, ac4, ac5, ac6 = result[:21]
                if Use_VAE:
                    _, rege_loss_value, latent_loss_value = result[:-3]
                for sen_id in range(BATCH_SIZE):
                    logits_total_prob += 1/scipy.stats.gmean((logits_value[sen_id].T)[batch[sen_id, 1:length[batch_i][sen_id]]].diagonal()
                                                             + 1e-6)/BATCH_SIZE
                # for i in range(4):
                #     _ = sess.run([d_train_op],feed_dict=feed_dict)

                d_total_loss_1 += d_loss_value_1
                d_total_loss_2 += d_loss_value_2
                g_total_loss_1 += g_loss_value_1
                g_total_loss_2 += g_loss_value_2
                pre_total_loss += pretrain_loss_value
                right_word_total_value += right_word_value
                total_gen_length += gen_length_value
                if batch_i % 10 == 9:
                    #valid_total_bleu += count_bleu(gen_sentence.tolist(), ch2int, valid_batches)
                    print('Epoch {:>3} Batch {:>4}/{} d_1 = {:.4f} d_2 = {:.4f}'
                          ' g_1 = {:.3f} g_2 = {:.3f} g_3 = {:.3f} pre = {:.3f} reco={:.3f} logits = {:.3f} right = {:.3f}'
                          ' gen_len = {:.3f} end_rate ={:.3f}'.format(
                        epoch, batch_i + 1, len(batches),
                        d_total_loss_1 / (batch_i + 1),d_total_loss_2 / (batch_i + 1),
                        g_total_loss_1 / (batch_i + 1),g_total_loss_2 / (batch_i + 1),g_loss_value_3,
                        pre_total_loss / (batch_i + 1),reco_loss_value,logits_total_prob/(batch_i + 1),
                        right_word_total_value/(batch_i + 1),total_gen_length/(batch_i + 1),
                        sess.run(end_rate)))
                    print(songnames[batch_i][0],' ',songnames[batch_i][1])
                    gen_sentence = gen_sentence.tolist()
                    output_process(gen_sentence[0], int2ch)
                    batch = batch.tolist()
                    output_process(batch[0], int2ch)
                    #print('')
                    print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} t={:.4f}'.format(
                        ac1,ac2,ac3,ac4,ac5,ac6,sess.run(temperature)))
                sess.run(tf.assign_add(end_rate, ((gen_length_value / expected_length - 1)) / 100))
                ac_val_1, ac_val_2, ac_val_3, ac_val_4, ac_val_5, ac_val_6 = \
                    ac_val_1 + ac1,ac_val_2+ ac2,ac_val_3 + ac3,ac_val_4+ac4,ac_val_5+ac5,ac_val_6+ac6

            if epoch % 1 == 0:
                save(tf.train.Saver(), sess, param.get_adv_save_dir(), epoch)
            print('Model Trained and Saved')
            #TODO:add compare bleu here
            avg_valid_bleu_1,avg_valid_bleu_2,avg_valid_bleu_3,avg_valid_bleu_4\
                = compare_bleu(num=1, sess=sess, write_json=False, generator=generator,batch_size=param.BATCH_SIZE)
            # gen_sentence = gen_sentence.tolist()
            # avg_valid_bleu = count_bleu(gen_sentence[:valid_batch_num],ch2int,valid_batches)
            avg_valid_prob,avg_valid_loss = valid(sess, logits, original_loss, embedding)

            bn = len(batches)
            string = 'Epoch {:>3} d_1 = {:.4f} d_2 = {:.4f} g_1 = {:.3f} g_2 = {:.3f} ' \
                     'pre_loss = {:.3f} valid_loss = {:.3f} valid_prob = {:.3f} Bleu = {:.3f},{:.3f},{:.3f},{:.3f} right = {:.3f}'.format(epoch,
                    d_total_loss_1/bn,d_total_loss_2/bn, g_total_loss_1/bn,g_total_loss_2/bn,
                    pre_total_loss/bn, avg_valid_loss, avg_valid_prob,avg_valid_bleu_1,avg_valid_bleu_2,avg_valid_bleu_3,
                    avg_valid_bleu_4,right_word_total_value/bn)\
                     +' '+' {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(ac_val_1/bn,ac_val_2/bn,ac_val_3/bn,ac_val_4/bn,ac_val_5/bn,ac_val_6/bn) \
                     + time.ctime()
            print(string)
            log.write(string+'\n')
            log.close()

def valid(sess,logits,loss,embedding):
    int2ch, ch2int = get_vocab(if_segment())
    dataloader = DataLoader()
    feature_dict = dataloader.get_audio_feature()
    batches, lengthes, features, songnames, lyrics = dataloader.get_batches(batch_size=BATCH_SIZE, set_no=2,
                                                                  feature_dict=feature_dict,
                                                                  lyric_dict=None)
    total_pro_value = 0
    total_pretrain_loss_value = 0
    batches = batches[0:valid_batch_num]
    for batch_i, batch in enumerate(batches):
        total_length = np.max(lengthes[batch_i])  # 一个batch中的最大长度
        gen_string = []
        feature_batch = features[batch_i]
        feature_batch, feature_length,song_label_value = prepare_feature_batch(feature_batch,songnames[batch_i], feature_dict)

        feed_dict = {encoder_inputs: feature_batch,
            encoder_lengths: feature_length,
            decoder_lengths: lengthes[batch_i],
            embed_ph: embedding,
            decoder_inputs: batch,
            song_label:song_label_value}
        if param.Use_lyric:
            lyric_batch = lyrics[batch_i]
            lyric_batch, lyric_length = prepare_lyric_batch(lyric_batch, ch2int)
            feed_dict[encoder_inputs_2] = lyric_batch
            feed_dict[encoder_lengths_2] = lyric_length
        logits_value,pretrain_loss_value = sess.run([logits,loss], feed_dict=feed_dict)
        for i in range(BATCH_SIZE):
            total_pro_value += 1 / scipy.stats.gmean((logits_value[i].T)[batch[i, 1:lengthes[batch_i][i]]].diagonal() + 1e-6) / BATCH_SIZE
        total_pretrain_loss_value += pretrain_loss_value
    return total_pro_value/len(batches),total_pretrain_loss_value/len(batches)

def valid_gan(sess,d_out_real,d_out_fake,generator,embedding):
    int2ch, ch2int = get_vocab(if_segment())
    dataloader = DataLoader()
    feature_dict = dataloader.get_audio_feature()
    max_len = tf.shape(decoder_inputs)[-1]
    encoder_final_state = generator.encoder_state_var()
    real_onehot = tf.one_hot(decoder_inputs[:, 0:max_len], param.VOCAB_SIZE, 1.0, 0.0)
    gen_o, gen_x, gen_x_onehot_adv = generator.generate_var_list()
    batches, lengthes, features, songnames, lyrics = dataloader.get_batches(batch_size=BATCH_SIZE, set_no=2,
                                                                  feature_dict=feature_dict,
                                                                  lyric_dict=None)
    total_pro_value = 0
    total_pretrain_loss_value = 0
    batches = batches[0:valid_batch_num]
    #d_out_real = discriminator_func(real_onehot,encoder_final_state, BATCH_SIZE, max_len, VOCAB_SIZE, embed_ph)
    #d_out_fake = discriminator_func(gen_x,encoder_final_state, BATCH_SIZE, max_len, VOCAB_SIZE, embed_ph)
    d_metric_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_real, labels=tf.ones_like(d_out_real)))
    d_metric_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_fake, labels=tf.ones_like(d_out_fake)))

    for batch_i, batch in enumerate(batches):
        total_length = np.max(lengthes[batch_i])  # 一个batch中的最大长度
        gen_string = []
        feature_batch = features[batch_i]
        feature_batch, feature_length,song_label_value = prepare_feature_batch(feature_batch,songnames[batch_i], feature_dict)

        feed_dict = {encoder_inputs: feature_batch,
            encoder_lengths: feature_length,
            decoder_lengths: lengthes[batch_i],
            embed_ph: embedding,
            decoder_inputs: batch,
            song_label:song_label_value}
        if param.Use_lyric:
            lyric_batch = lyrics[batch_i]
            lyric_batch, lyric_length = prepare_lyric_batch(lyric_batch, ch2int)
            feed_dict[encoder_inputs_2] = lyric_batch
            feed_dict[encoder_lengths_2] = lyric_length
        d_real_value,d_fake_value = sess.run([d_metric_real,d_metric_fake], feed_dict=feed_dict)

    return d_real_value,d_fake_value





#pretrain(5)
adv_train(50)
