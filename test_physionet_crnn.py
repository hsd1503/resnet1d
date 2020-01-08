"""
test on physionet data

Shenda Hong, Nov 2019
"""

import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from util import read_data_physionet_2, read_data_physionet_2_clean, read_data_physionet_4, preprocess_physionet
from crnn1d import CRNN, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary

if __name__ == "__main__":

    is_debug = False
    
    batch_size = 128
    if is_debug:
        writer = SummaryWriter('/nethome/shong375/log/crnn1d/challenge2017/debug')
    else:
        writer = SummaryWriter('/nethome/shong375/log/crnn1d/challenge2017/new_novote_cleandata_rerunx')

    # make data
    # preprocess_physionet() ## run this if you have no preprocessed data yet
    X_train, X_test, Y_train, Y_test, pid_test = read_data_physionet_2_clean(window_size=3000, stride=500)
    print(X_train.shape, Y_train.shape)
    dataset = MyDataset(X_train, Y_train)
    dataset_test = MyDataset(X_test, Y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)
    
    # make model
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = CRNN(
        in_channels=1, 
        out_channels=128, 
        n_len_seg=50, 
        verbose=True,
        n_classes=2,
        device=device)
    model.to(device)

    ## look model
    prog_iter = tqdm(dataloader, desc="init", leave=False)
    for batch_idx, batch in enumerate(prog_iter):
        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)
        break

    # train and test
    model.verbose = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()

    n_epoch = 200
    step = 0
    prev_f1 = 0
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):

        # train
        model.train()
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            writer.add_scalar('Loss/train', loss.item(), step)

            if is_debug:
                break
        
        scheduler.step(_)
                    
        # test
        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                all_pred_prob.append(pred.cpu().data.numpy())
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)
        ## classification report
        tmp_report = classification_report(Y_test, all_pred, output_dict=True)
        print(confusion_matrix(Y_test, all_pred))
        f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'])/2
        if f1_score > prev_f1:
            torch.save(model, 'models/crnn/cleandata/crnn_{}_{:.4f}.pt'.format(_, f1_score))
            prev_f1 = f1_score
        writer.add_scalar('F1/f1_score', f1_score, _)
        writer.add_scalar('F1/label_0', tmp_report['0']['f1-score'], _)
        writer.add_scalar('F1/label_1', tmp_report['1']['f1-score'], _)
