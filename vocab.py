import os
import json
import segment
import param
from param import *
import codecs
import re
VOCAB_SIZE  = 100000
data_dir = param.get_data_dir()
_vocab_path = os.path.join(data_dir, 'vocab.json')

def _gen_vocab(if_segment = False):
    data_dir = param.get_data_dir()
    songlist = os.listdir(data_dir)
    corpus,_ = get_corpus(if_segment)
    char_cnts = dict()
    for sentence in corpus:
        for ch in sentence:
            char_cnts[ch] = char_cnts[ch] + 1 if ch in char_cnts else 1
    vocab = sorted([ch for ch in char_cnts], key=lambda ch: -char_cnts[ch])[:VOCAB_SIZE - 2]
    with codecs.open(_vocab_path, 'w', 'utf-8') as fout:
        json.dump(vocab, fout)
    print(1)

def get_vocab(if_segment = False):
    if not os.path.exists(_vocab_path):
        _gen_vocab(if_segment)
    int2ch = []
    with codecs.open(_vocab_path, 'r', 'utf-8') as fin:
        int2ch.extend(json.load(fin))
    ch2int = dict((ch, idx) for idx, ch in enumerate(int2ch))
    return int2ch, ch2int


# set_no =0:全部，1:训练集，2:验证集，3:测试集
def get_corpus(if_segment = False,set_no = 0,songidx = None):#返回句子，二维列表
    dir = param.get_data_dir()
#    stop_dir = param.get_stop_dir()
#    stopwords = [line.strip() for line in (open(stop_dir, 'r', encoding='utf-8')).readlines()]
    stopwords = ['@','#','、','~','♡','“','”']
    allowed_words = ['.',',','?','!','\\','-','n',';',' ']
    songlist = os.listdir(dir)
    try:
        songlist.remove('vocab.json')
        songlist.remove('word2vec.npy')
    except:
        pass
    if param.divide_type == 1:
        songlist_len = len(songlist)
        if set_no == 0:
            songlist = songlist
        elif set_no == 1:
            new_songlist = []
            for song_id in range(songlist_len):
                if song_id % 10 < 8:
                    new_songlist.append(songlist[song_id])
            songlist = new_songlist
        elif set_no == 2:
            new_songlist = []
            for song_id in range(songlist_len):
                if song_id % 10 == 8:
                    new_songlist.append(songlist[song_id])
            songlist = new_songlist
        elif set_no == 3:
            new_songlist = []
            for song_id in range(songlist_len):
                if song_id % 10 > 8:
                    new_songlist.append(songlist[song_id])
            songlist = new_songlist

    corpus = []
    song_names = []
    if songidx is not None:
        songlist = songlist[songidx:songidx + 1]
    for name in songlist:
        temp_corpus = []
        temp_song_names = []
        sentance_keyword_dict = {}
        songdir = dir + '/' + name

        try:
            with open(songdir + '/' + name + '_hot.json', 'r', encoding='utf-8') as file:
                text = json.load(file)
                processed_hot_text = text_preprocess(text)
            for sentance_seg2 in processed_hot_text:
                if len(sentance_seg2) > sentence_min_len:
                    if Use_remove_duplicate:
                        if ''.join(sentance_seg2[1:-1]) not in sentance_keyword_dict:
                            sentance_keyword_dict[''.join(sentance_seg2[1:-1])] = 1
                        else:
                            continue
                    temp_corpus.append(sentance_seg2)
                    temp_song_names.append(name)
            if param.divide_type == 2:
                temp_corpus,temp_song_names = divide_dataset(temp_corpus,temp_song_names,set_no)
                import copy
            for i in range(hot_copy_times):
                corpus.extend(copy.deepcopy(temp_corpus))
                song_names.extend(copy.deepcopy(temp_song_names))

            temp_corpus = []
            temp_song_names = []
            with open(songdir + '/' + name + '.json', 'r', encoding='utf-8') as file:
                text = json.load(file)
                processed_text = text_preprocess(text)
            for sentance_seg2 in processed_text:
                if len(sentance_seg2) > sentence_min_len:
                    if Use_remove_duplicate:
                        if ''.join(sentance_seg2[1:-1]) not in sentance_keyword_dict:
                            sentance_keyword_dict[''.join(sentance_seg2[1:-1])] = 1
                        else:
                            continue
                    temp_corpus.append(sentance_seg2)
                    temp_song_names.append(name)
            if param.divide_type == 2:
                temp_corpus,temp_song_names = divide_dataset(temp_corpus,temp_song_names,set_no)
            corpus.extend(temp_corpus.copy())
            song_names.extend(temp_song_names.copy())
        except:
            print('error in vocab')
            continue
    return corpus,song_names

def divide_dataset(temp_corpus,temp_song_names,set_no):
    text_number = len(temp_corpus)
    if set_no == 0:
        temp_corpus = temp_corpus
        temp_song_names = temp_song_names
    elif set_no == 1:
        temp_corpus = temp_corpus[0:int(text_number * 0.8)]
        temp_song_names = temp_song_names[0:int(text_number * 0.8)]
    elif set_no == 2:
        temp_corpus = temp_corpus[int(text_number * 0.8): int(text_number * 0.9)]
        temp_song_names = temp_song_names[int(text_number * 0.8): int(text_number * 0.9)]
    elif set_no == 3:
        temp_corpus = temp_corpus[int(text_number * 0.9):]
        temp_song_names = temp_song_names[int(text_number * 0.9):]
    return temp_corpus,temp_song_names

def is_chinese(word):
    for uchar in word:
        """判断一个unicode是否是汉字"""
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
            return True
    else:
        return False



def text_preprocess(text):
    allowed_words = ['.',',','?','!','\\','-','n',';',' ','W']
    for sen_id,sentance in enumerate(text):
        sentance = (sentance).replace('\n', '.')
        sentance = (sentance).replace('\t', '.')
        sentance = (sentance).replace('。', '.')
        sentance = (sentance).replace('，', ',')
        sentance = (sentance).replace('？', '?')
        text[sen_id] = sentance
    pattern1 = re.compile('[a-zA-Z]+')
    pattern2 = re.compile(r'\[.*?\]')
    pattern3 = re.compile(r'\.+')
    pattern4 = re.compile(r' +')
    pattern5 = re.compile(r'…+')
    pattern6 = re.compile(r'\\r')
    pattern7 = re.compile('[0-9]+')
    pattern8 = re.compile(r',+')
    pattern9 = re.compile(r'《.*?》')
    pattern10 = re.compile(r'。+')
    pattern11 = re.compile(r'，+')
    pattern12 = re.compile(r'【.*?】')
    text = [pattern1.sub("", lines) for lines in text]  # 去掉英语字符
    text = [pattern2.sub("", lines) for lines in text]  # 去掉[]中的部分
    text = [pattern3.sub(".", lines) for lines in text]  # 去掉...
    text = [pattern4.sub(".", lines) for lines in text]  # 去掉空格
    text = [pattern5.sub(".", lines) for lines in text]  # 去掉…
    text = [pattern6.sub("", lines) for lines in text]  # 去掉空格\r
    text = [pattern7.sub("n", lines) for lines in text]  # 用n代表数字
    text = [pattern8.sub(",", lines) for lines in text]  # 去掉,,,
    text = [pattern9.sub("", lines) for lines in text]  # 去掉《》中的部分
    text = [pattern10.sub(".", lines) for lines in text]  # 去掉。。。
    text = [pattern11.sub(",", lines) for lines in text]  # 去掉，，，
    text = [pattern12.sub("", lines) for lines in text]  # 去掉【】中
    text = [pattern3.sub(".", lines) for lines in text]  # 去掉...
    text_id = 0
    max_len = 20
    processed_text = []
    while (True):
        if text_id >= text.__len__():
            break
        if text[text_id].__len__() <= max_len:
            text_id += 1
            continue
        for i in range(text[text_id].__len__() - max_len):
            if text[text_id][i + max_len] in [',', '.','?']:
                text.append(text[text_id][i + max_len + 1:])
                text[text_id] = text[text_id][:i + max_len]
                text_id += 1
                break
            if i == text[text_id].__len__() - max_len -1:
                text_id += 1
                break
            if i == sentence_min_len:
                break_sign = 0
                for j in range(max_len - 1,-1,-1):
                    if text[text_id][j] in [',', '.', '。', '，', '\n']:
                        text.append(text[text_id][j+1:])
                        text[text_id] = text[text_id][:j+1]
                        text_id += 1
                        break_sign = 1
                        break
                if break_sign:
                    break
                else:
                    # text.append(text[text_id][i + max_len + 1:])
                    text[text_id] = text[text_id][:i + max_len + 1]
                    text_id += 1
                    break

    for sentance in text:
        if if_segment():
            sentance_seg = segment.seg(sentance)
        else:
            sentance_seg = sentance
        sentance_seg2 = ['\start']
        for word in sentance_seg:
            if word in allowed_words or is_chinese(word):
                sentance_seg2.append(word)
        sentance_seg2.append('\end')
        processed_text.append(sentance_seg2)
    return processed_text


if __name__ == '__main__':
    get_vocab(if_segment= param.if_segment())
    songlist = get_corpus()
