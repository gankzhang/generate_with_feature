import os
def get_data_dir():
    data_dir = './data/mix_715'
    return data_dir
def get_audio_dir():
    return './data/for_audio_10'
def get_save_dir():
    return './data/model/rnn/'
def get_adv_save_dir():
    return './data/model/adv/'
def get_generate_text_dir():
    return './data/generate_text/'
def if_segment():
    return False
def wavenet_param_dir():
    return './data/wavenet_params.json'

hot_copy_times = 10
sentence_min_len = 10


NUM_UNITS = 64
INPUT_DIM = 128
NUM_LAYERS = 1
BATCH_SIZE = 8
LOSS_TYPE = 2
LEARNRATE = 0.01
decay_rate = 0.97
N_EPOCH = 50

#VOCAB_SIZE = 5889
VOCAB_SIZE = 6694
temperature = 10

start_length = 0
Songnum = 83
divide_type = 2
#relational memory
mem_slots = 1
head_size = 128#changed from 512
num_heads = 2
gan_type = 'standard'
D_with_state = True


Use_lyric = False
Use_VAE = False
Use_relational_memory = True
Use_remove_duplicate= True
Use_latent_z = True
Use_Weight = False
Post_with_state = False

expected_length = 16
valid_batch_num = 5
adv_train_batch_num = 1000
adv_start_step = 5
n_adv_step = 20
bleu_weights = [1/3,1/3,1/3,0]
d_lr_rate = 0.01
g_lr_rate = 0.01
pre_lr_rate = 0.01
d_2_rate = 10
