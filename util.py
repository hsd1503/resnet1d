import numpy as np


def read_data():
    n_samples = 1000
    n_length = 1000
    n_channel = 12
    n_class = 3
    data = np.random.rand(n_samples, n_channel, n_length)
    label = np.random.randint(0, n_class, n_samples)
    return data, label