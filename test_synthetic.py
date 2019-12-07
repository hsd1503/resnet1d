"""
test on synthetic data

Shenda Hong, Oct 2019
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 

from util import read_data_generated
from resnet1d import ResNet1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

    
if __name__ == "__main__":
    
    # make data
    n_samples = 1000
    n_length = 2048
    n_channel = 18
    n_classes = 6
    data, label = read_data_generated(n_samples=n_samples, n_length=n_length, n_channel=n_channel, n_classes=n_classes)
    print(data.shape, Counter(label))
    dataset = MyDataset(data, label)
    dataloader = DataLoader(dataset, batch_size=64)
    
    # make model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## change the hyper-parameters for your own data
    # (n_block, downsample_gap, increasefilter_gap) = (8, 1, 2)
    # 34 layer (16*2+2): 16, 2, 4
    # 98 layer (48*2+2): 48, 6, 12
    model = ResNet1D(
        in_channels=n_channel, 
        base_filters=128, 
        kernel_size=16, 
        stride=2, 
        n_block=48, 
        groups=32,
        n_classes=n_classes, 
        downsample_gap=6, 
        increasefilter_gap=12, 
        verbose=False)
    model.to(device)
    summary(model, (data.shape[1], data.shape[2]))
    exit()


    # train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    all_loss = []
    prog_iter = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(prog_iter):

        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)

        loss = loss_func(pred, input_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_loss.append(loss.item())
    
    plt.plot(all_loss)
    
    # test
    data_test, label_test = read_data_generated(n_samples=n_samples, n_length=n_length, n_channel=n_channel, n_classes=n_classes)
    print(data_test.shape, Counter(label_test))
    dataset_test = MyDataset(data_test, label_test)
    dataloader_test = DataLoader(dataset_test, batch_size=64, drop_last=False)
    prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
    all_pred_prob = []
    for batch_idx, batch in enumerate(prog_iter_test):
        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)
        all_pred_prob.append(pred.cpu().data.numpy())
    all_pred_prob = np.concatenate(all_pred_prob)
    all_pred = np.argmax(all_pred_prob, axis=1)
    ## classification report
    print(classification_report(all_pred, label_test))
    
    
    
    