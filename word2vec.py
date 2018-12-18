from gensim import models
from vocab import *
import vocab
import numpy as np
from numpy.random import uniform

# embedding layer的大小
data_dir = param.get_data_dir()
_w2v_path = os.path.join(data_dir, 'word2vec.npy')
ndim = param.get_embed_dim()

def gen_embedding(if_segment = False):
    int2ch, ch2int = get_vocab(if_segment)
    corpus,_ = vocab.get_corpus(if_segment)
    model = models.Word2Vec(corpus, size=ndim,min_count=1)
    embedding = uniform(-1.0, 1.0, [len(int2ch), ndim])
    for idx, ch in enumerate(int2ch):
        if ch in model.wv:
            embedding[idx, :] = model.wv[ch]
    np.save(_w2v_path, embedding)
    print("Word embedding is saved.")



gen_embedding(if_segment= param.if_segment())