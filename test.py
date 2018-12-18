import tensorflow as tf
from tensorflow.contrib import rnn
from generate import *


def test_train(generator = None):
    saver = tf.train.Saver()
    embedding = np.load(os.path.join(data_dir, 'word2vec.npy'))
    with tf.Session() as sess:
        generator._init_vars(sess)
        load(saver, sess, save_dir)
        print('Model Loaded')
        batches, length = generator.get_batches(BATCH_SIZE, set_no=1)
        total_loss = 0
        for batch_i, batch in enumerate(batches):
            target_batch = generator.get_target_batch(batch,length[batch_i])
            outputs, loss = sess.run([generator.decoder_final_state, generator.loss], feed_dict={
                generator.decoder_inputs: batch[:, :-1],
                generator.targets: target_batch[:, 1:],
                generator.decoder_lengths: length[batch_i],
                generator._embed_ph: embedding})
            total_loss += loss
#            print('Batch {:>4}/{}   train_loss = {:.3f}'.format(batch_i,len(batches),loss))
        print('Valid Total loss = {:.3f}'.format(total_loss))

def valid(generator = None):
    saver = tf.train.Saver()
    embedding = np.load(os.path.join(data_dir, 'word2vec.npy'))
    with tf.Session() as sess:
        generator._init_vars(sess)
        load(saver, sess, save_dir)
        batches, lengthes = generator.get_batches(1, set_no=2)
        start_word = 10
        total_loss = 0
        for batch_i, batch in enumerate(batches):
            total_length = lengthes[batch_i][0]
            if total_length < start_word:
                continue
            prime_word = batch[:][0:start_word].tolist()
            gen_sentence = []
            for i in range(BATCH_SIZE):
                gen_sentence.append([])
                gen_sentence[i].extend(prime_word[i])

            for length in range(start_word-1, total_length):
                if length == start_word - 1:
                    input_length = start_word * np.ones(BATCH_SIZE)
                    state = sess.run(generator.decoder_init_state)
                    prob, state = sess.run([generator.probs, generator.decoder_final_state], feed_dict={
                        generator.decoder_inputs: np.array(gen_sentence),
                        generator.decoder_init_state: state,
                        generator.decoder_lengths: input_length,
                        generator._embed_ph: embedding})
                    continue
                else:
                    input_length = np.ones(BATCH_SIZE)
                    prob, state = sess.run([generator.probs, generator.decoder_final_state], feed_dict={
                        generator.decoder_inputs: np.array(gen_sentence)[:, -1:],
                        generator.decoder_init_state: state,
                        generator.decoder_lengths: input_length,
                        generator._embed_ph: embedding})
                temp = prob[0,length - 1].tolist()
                pre_word = pick_word(temp, generator.int2ch)
                for i in range(BATCH_SIZE):
                    gen_sentence[i].append(pre_word)

            input_length = total_length * np.ones(BATCH_SIZE)
            outputs,valid_loss = sess.run([generator.decoder_final_state, generator.loss], feed_dict={
                    generator.decoder_inputs: np.array(gen_sentence)[:,:-1],
                    generator.targets: np.array(gen_sentence)[:,1:],
                    generator.decoder_lengths:input_length,
                    generator._embed_ph: embedding})
            total_loss += valid_loss
            print('Batch {:>4}/{}   valid_loss = {:.3f}'.format(
                batch_i,
                len(batches),
                valid_loss))
            for word_i, word in enumerate(gen_sentence[0]):
                gen_sentence[0][word_i] = generator.int2ch[word]
            print(''.join(gen_sentence[0]))
        print(total_loss)
    return 1

def test():
    return 1

if __name__ == '__main__':
    generator = Generator()
    generator.generate(10,start_word=10,write_json= True)
#    test_train(generator = generator)
#    valid(generator = generator)
