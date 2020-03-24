"""
cnn+rnn for 1-d signal data, pytorch version
 
Shenda Hong, Jan 2020
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)

class CRNN(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, out_channels, n_len_seg, n_classes, device, verbose=False):
        super(CRNN, self).__init__()
        
        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.device = device
        self.verbose = verbose

        # (batch, channels, length)
        self.cnn = nn.Conv1d(in_channels=self.in_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=16, 
                            stride=2)
        # (batch, seq, feature)
        self.rnn = nn.LSTM(input_size=(self.out_channels), 
                            hidden_size=self.out_channels, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=False)
        self.dense = nn.Linear(out_channels, n_classes)
        
    def forward(self, x):

        self.n_channel, self.n_length = x.shape[-2], x.shape[-1]
        assert (self.n_length % self.n_len_seg == 0), "Input n_length should divided by n_len_seg"
        self.n_seg = self.n_length // self.n_len_seg

        out = x
        if self.verbose:
            print(out.shape)

        # (n_samples, n_channel, n_length) -> (n_samples, n_length, n_channel)
        out = out.permute(0,2,1)
        if self.verbose:
            print(out.shape)
        # (n_samples, n_length, n_channel) -> (n_samples*n_seg, n_len_seg, n_channel)
        out = out.view(-1, self.n_len_seg, self.n_channel)
        if self.verbose:
            print(out.shape)
        # (n_samples*n_seg, n_len_seg, n_channel) -> (n_samples*n_seg, n_channel, n_len_seg)
        out = out.permute(0,2,1)
        if self.verbose:
            print(out.shape)
        # cnn
        out = self.cnn(out)
        if self.verbose:
            print(out.shape)
        # global avg, (n_samples*n_seg, out_channels)
        out = out.mean(-1)
        if self.verbose:
            print(out.shape)
        out = out.view(-1, self.n_seg, self.out_channels)
        if self.verbose:
            print(out.shape)
        _, (out, _) = self.rnn(out)
        out = torch.squeeze(out, dim=0)
        if self.verbose:
            print(out.shape)
        out = self.dense(out)
        if self.verbose:
            print(out.shape)
        
        return out
