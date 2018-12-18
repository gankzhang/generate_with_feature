import numpy as np
import random


# 不知道为什么作为class Generator 的函数就会出错..
def pick_word(probabilities,int2ch):#return int
    probabilities[3]/=2
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
    return ch