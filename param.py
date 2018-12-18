import os
def get_data_dir():
    data_dir = './data/纯音乐的殿堂_900_918'
    return data_dir
def get_stop_dir():
    stop_dir = './data/stopwords.txt'
    return stop_dir
def get_audio_dir():
#    return './for_colab_2/data/适合咖啡厅、阅读的安静纯音乐'#/A Little Story'
    return './data/for_audio_10'
def get_save_dir():
#    return os.path.join('./data/model/rnn/','model.ckpt')
    return './data/model/rnn/'
def get_generate_text_dir():
    return './data/generate_text/'
def get_embed_dim():
    return 128
def if_segment():
    return False
def wavenet_param_dir():
    return './data/wavenet_params.json'