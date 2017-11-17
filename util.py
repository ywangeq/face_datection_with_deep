import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from model import eval_in_batches


num_epochs =1000
ealry_stop_patience = 100
image_size = 96    # face image
validation_size =100 # number of samples in validation dataset
batch_size =64       #number of samples in a training mini-batch
eval_batch_size = 64 #number of samples in intermedia batch for performance chech per epoch
num_channels =3
num_labels = 30
seed = None
debug_mode =True
Dropout_Rate = 0.5
L2_reg_peram = 1e-7
learning_base_rate = 1e-3
learning_decay_rate =0.95
adam_reg_peram = 0.95

def load(filename,test = False):
    """

    :param filename:
    :param test:
    load the face_keypoints detection csv data into numpy structures from pandas

    :return: (x,y) tuple of (num_image,image_size,image_size,1) shaped the image tensor
    """
    dataframe = pd.read_csv(filename)
    feature_cols = dataframe.columes[:-1] #image colmn
    dataframe['Image'] = dataframe['Image'].apply(lambda img:np.fromstring(img,sep=' ')/255.0)
    dataframe =dataframe.dropna()

    X = np.vstack(dataframe['Image'])
    X = X.reshape(-1,image_size,image_size,1)

    if not test:
        y=dataframe[feature_cols].values/96.0
        X,y = shuffle(X,y)
    else :
        y =None
    return X,y

def generate_test_results(test_data,sess,eval_prediction,eval_data_node):
    test_labels = eval_in_batches(test_data,sess,eval_prediction,eval-eval_data_node)

    test_labels*=96.0
    test_labels =test_labels.clip(0,96)

    results = pd.DataFrame(test_labels,columns=('ImageNumber','FeatureVector'))
    results.to._csv('result_vector.csv',index=False)
    dprint("wrote test result to result_vectors.csv")

def dprint(obj):
    if debug_mode:
        print(obj)