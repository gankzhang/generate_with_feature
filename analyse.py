from data_utils import *
from vocab import get_corpus
from jieba import analyse
textrank = analyse.textrank
total_song_num = 0
temp_vocab = np.zeros([VOCAB_SIZE])
int2ch, ch2int = get_vocab(if_segment())

keywords = {}
total_total_length =0
total_total_count = 0
len_max = 0
len_min = 100000000
for song_id in range(83):
    total_text, song_names = get_corpus(if_segment=False, set_no=0,songidx=song_id)
    max_len = -1
    word_num = 0
    total_length = 0
    for text in total_text:
        total_length += text.__len__() -2
    print(total_length/len(total_text),len(total_text))
    total_total_length += total_length
    total_total_count += len(total_text)
    len_max = max(len(total_text),len_max)
    len_min = min(len(total_text),len_min)

print(total_total_length/total_total_count,total_total_count,len_max,len_min)
        # for ch in text[1:]:
        #     temp_vocab[ch2int[ch]] += 1

        # print(''.join(text[1:-1]))
#        max_len = max(max_len,text.__len__())
#         keyword = textrank(''.join(text[1:-1]))
#         for word in keyword:
#             if word in keywords.keys():
#                 keywords[word] += 1
#             else:
#                 keywords[word] = 1
#             word_num += 1
#     for key in keywords.keys():
#         keywords[key]/= word_num
# keywords = sorted(keywords.items(),key = lambda item:item[1],reverse = True)
# for keyword in keywords:
#     print(keyword)

#     print(np.sum(temp_vocab))
# temp_vocab = temp_vocab/np.sum(temp_vocab)
# a = np.dot(temp_vocab,temp_vocab)
# print(a)
    # total_song_num += total_text.__len__()
    # print(song_id,total_text.__len__(),total_song_num,max_len,song_names[0])
