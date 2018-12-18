


from vocab import *




def batch_train_data(batch_size):
#    if not os.path.exists(train_path):
#        _gen_train_data()
    _, ch2int = get_vocab()
    with codecs.open(train_path, 'r', 'utf-8') as fin:
        stop = False
        while not stop:
            batch_s = [[] for _ in range(4)]
            batch_kw = [[] for _ in range(4)]
            for i in range(batch_size):
                line = fin.readline()
                if not line:
                    stop = True
                    break
                else:
                    toks = line.strip().split('\t')
                    batch_s[i%4].append([0]+[ch2int[ch] for ch in toks[0]]+[VOCAB_SIZE-1])
                    batch_kw[i%4].append([ch2int[ch] for ch in toks[1]])
            if 0 == len(batch_s[0]):
                break
            else:
                kw_mats = [fill_np_matrix(batch_kw[i], batch_size, VOCAB_SIZE-1) \
                        for i in range(4)]
                kw_lens = [fill_np_array(map(len, batch_kw[i]), batch_size, 0) \
                        for i in range(4)]
                s_mats = [fill_np_matrix(batch_s[i], batch_size, VOCAB_SIZE-1) \
                        for i in range(4)]
                s_lens = [fill_np_array([len(x)-1 for x in batch_s[i]], batch_size, 0) \
                        for i in range(4)]
                yield kw_mats, kw_lens, s_mats, s_lens