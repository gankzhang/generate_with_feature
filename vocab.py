import os
import json
import segment
import param
import codecs
import re
VOCAB_SIZE  = 20000
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
def get_corpus(if_segment = False,set_no = 0):#返回句子，二维列表
    dir = param.get_data_dir()
#    stop_dir = param.get_stop_dir()
#    stopwords = [line.strip() for line in (open(stop_dir, 'r', encoding='utf-8')).readlines()]
    stopwords = ['@','#','、','~','♡','“','”']
    songlist = os.listdir(dir)
    try:
        songlist.remove('vocab.json')
        songlist.remove('word2vec.npy')
    except:
        pass
    songlist_len = len(songlist)
    if set_no == 0:
        songlist = songlist
    elif set_no == 1:
        songlist = songlist[0:int(songlist_len * 0.6)]
    elif set_no == 2:
        songlist = songlist[int(songlist_len * 0.6) : int(songlist_len * 0.8)]
    elif set_no == 3:
        songlist = songlist[int(songlist_len * 0.8):]
    corpus = []
    song_names = []
    pattern1 = re.compile('[a-zA-Z0-9]+')
    pattern2 = re.compile(r'\[.*?\]')
    pattern3 = re.compile(r'\.+')
    pattern4 = re.compile(r' +')
    pattern5 = re.compile(r'…+')
    pattern6 = re.compile(r'\\r')
    pattern7 = re.compile('[0-9]+')
#    pattern7 = re.compile(r'。+')
    for name in songlist:
        songdir = dir + '/' + name
        try:
            with open(songdir + '/' + name + '.json', 'r', encoding='utf-8') as file:
                text = json.load(file)
                text = [pattern1.sub("", lines) for lines in text]  # 去掉英语字符
                text = [pattern2.sub("", lines) for lines in text]  # 去掉[]中的部分
                text = [pattern3.sub(".", lines) for lines in text]#去掉...
                text = [pattern4.sub(",", lines) for lines in text]#去掉空格
                text = [pattern5.sub(".", lines) for lines in text]#去掉…
                text = [pattern6.sub("", lines) for lines in text]#去掉空格\r
                text = [pattern7.sub('/num', lines) for lines in text]  # 数字变成 \num
                #               text = [pattern7.sub("。", lines) for lines in text]  # 去掉。。。


                for sentance in text:
                    sentance_seg = []
                    if if_segment:
                        sentance_seg = segment.seg(sentance)
                    else:
                        sentance = (sentance).replace('\n','。')
                        sentance = (sentance).replace('\t', '，')
                        sentance = (sentance).replace('。','.')
                        sentance = (sentance).replace('，',',')
                        sentance_seg = sentance
                    sentance_seg2= ['\start']
                    for word in sentance_seg:
                        if word not in stopwords:
                            sentance_seg2.append(word)
 #                           sentance_seg2.append(word)
                    sentance_seg2.append('\end')
                    if len(sentance_seg2) > 15:
                        corpus.append(sentance_seg2)
                        song_names.append(name)
        except:
            continue
    return corpus,song_names


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
if __name__ == '__main__':
#    get_vocab(if_segment= param.if_segment())
    songlist = get_corpus()
