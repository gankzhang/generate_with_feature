import jieba
def seg(text):
    return jieba.lcut(text, HMM=True)