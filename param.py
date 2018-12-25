import os
def get_data_dir():
    data_dir = './data/纯音乐的殿堂_900_918_1'
    return data_dir
def get_stop_dir():
    stop_dir = './data/stopwords.txt'
    return stop_dir
def get_audio_dir():
    return './data/for_audio_10'
def get_save_dir():
    return './data/model/rnn/'
def get_generate_text_dir():
    return './data/generate_text/'
def get_embed_dim():
    return 256
def if_segment():
    return False
def wavenet_param_dir():
    return './data/wavenet_params.json'