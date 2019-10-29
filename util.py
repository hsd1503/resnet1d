import numpy as np


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
