from param import *
import numpy as np
def get_audio_feature():
    files = os.listdir(get_data_dir())
    feature = dict()
    for file in files:
        try:
            temp = np.load(get_data_dir()+'/'+file+'/'+file+'.npy')
            feature[file] = temp
        except:
            pass
    print('audio feature read')
    return feature

#feature = get_audio_feature()
#print('1')