import numpy as np
import pandas as pd
import scipy.io
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

def preprocess_physionet():
    """
    download the raw data from https://physionet.org/content/challenge-2017/1.0.0/
    """
    
    # read label
    label_df = pd.read_csv('challenge2017/REFERENCE-v3.csv', header=None)
    label = label_df.iloc[:,1].values
    print(Counter(label))

    # read data
    all_data = []
    filenames = pd.read_csv('challenge2017/training2017/RECORDS', header=None)
    filenames = filenames.iloc[:,0].values
    print(filenames)
    for filename in tqdm(filenames):
        mat = scipy.io.loadmat('challenge2017/training2017/{0}.mat'.format(filename))
        mat = np.array(mat['val'])[0]
        all_data.append(mat)
    all_data = np.array(all_data)

    res = {'data':all_data, 'label':label}
    with open('challenge2017/challenge2017.pkl', 'wb') as fout:
        pickle.dump(res, fout)

def slide_and_cut(X, Y, window_size, stride, output_pid=False):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            i_stride = stride//6 # use 10 for read_data_physionet_2
        elif tmp_Y == 2:
            i_stride = stride//2
        elif tmp_Y == 3:
            i_stride = stride//20
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)

def read_data_physionet_2(window_size=3000, stride=500):

    # read pkl
    with open('../data/challenge2017/challenge2017.pkl', 'rb') as fin:
        res = pickle.load(fin)
    ## scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std
    all_data = res['data']
    ## encode label
    all_label = []
    for i in res['label']:
        if i == 'A':
            all_label.append(1)
        else:
            all_label.append(0)
    all_label = np.array(all_label)

    # split train test
    X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.1, random_state=0)
    
    # slide and cut
    print('before: ')
    print(Counter(Y_train), Counter(Y_test))
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_test))
    
    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    return X_train, X_test, Y_train, Y_test, pid_test

def read_data_physionet_4(window_size=3000, stride=500):

    # read pkl
    with open('../data/challenge2017/challenge2017.pkl', 'rb') as fin:
        res = pickle.load(fin)
    ## scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std
    ## encode label
    all_label = []
    for i in res['label']:
        if i == 'N':
            all_label.append(0)
        elif i == 'A':
            all_label.append(1)
        elif i == 'O':
            all_label.append(2)
        elif i == '~':
            all_label.append(3)
    all_label = np.array(all_label)

    # split train test
    X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.1, random_state=0)
    
    # slide and cut
    print('before: ')
    print(Counter(Y_train), Counter(Y_test))
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_test))
    
    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    return X_train, X_test, Y_train, Y_test, pid_test


def read_data_generated(n_samples, n_length, n_channel, n_classes, verbose=False):
    """
    Generated data
    
    This generated data contains one noise channel class, plus unlimited number of sine channel classes which are different on frequency. 
    
    """    
    all_X = []
    all_Y = []
    
    # noise channel class
    X_noise = np.random.rand(n_samples, n_channel, n_length)
    Y_noise = np.array([0]*n_samples)
    all_X.append(X_noise)
    all_Y.append(Y_noise)
    
    # sine channel classe
    x = np.arange(n_length)
    for i_class in range(n_classes-1):
        scale = 2**i_class
        offset_list = 2*np.pi*np.random.rand(n_samples)
        X_sin = []
        for i_sample in range(n_samples):
            tmp_x = []
            for i_channel in range(n_channel):
                tmp_x.append(np.sin(x/scale+2*np.pi*np.random.rand()))
            X_sin.append(tmp_x)
        X_sin = np.array(X_sin)
        Y_sin = np.array([i_class+1]*n_samples)
        all_X.append(X_sin)
        all_Y.append(Y_sin)

    # combine and shuffle
    all_X = np.concatenate(all_X)
    all_Y = np.concatenate(all_Y)
    shuffle_idx = np.random.permutation(all_Y.shape[0])
    all_X = all_X[shuffle_idx]
    all_Y = all_Y[shuffle_idx]
    
    # random pick some and plot
    if verbose:
        for _ in np.random.permutation(all_Y.shape[0])[:10]:
            fig = plt.figure()
            plt.plot(all_X[_,0,:])
            plt.title('Label: {0}'.format(all_Y[_]))
    
    return all_X, all_Y
