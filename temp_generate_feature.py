import numpy as np
import os
import param
# data = np.load('feature_716.npy')
# songlist = os.listdir(param.get_data_dir())
# songlist.remove('vocab.json')
# songlist.remove('word2vec.npy')
# songlist.sort()
# for song_id,song in enumerate(songlist):
#     feature = data[song_id]
#     feature = np.reshape(feature,[2050,-1,64])
#     np.save(param.get_data_dir()+'/'+song+'/'+song+'.npy',feature)

songlist = os.listdir('./human_score/data')
# songlist.remove('vocab.json')
# songlist.remove('word2vec.npy')
import shutil
for song in songlist:
    shutil.copyfile('./human_score/data/'+song+'/'+song+'.mp3','./human_score/data/'+song+'.mp3')


    # if os.path.exists('./human_score/data/'+song+'/'+song+'.npy'):
    #     os.remove('./human_score/data/'+song+'/'+song+'.npy')
    # if os.path.exists('./human_score/data/'+song+'/'+song+'.json'):
    #     os.remove('./human_score/data/'+song+'/'+song+'.json')
    # if os.path.exists('./human_score/data/'+song+'/'+song+'_hot.json'):
    #     os.remove('./human_score/data/'+song+'/'+song+'_hot.json')
    # if os.path.exists('./human_score/data/'+song+'/'+song+'_lyric.txt'):
    #     os.remove('./human_score/data/'+song+'/'+song+'_lyric.txt')