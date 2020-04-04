"""
test on physionet data

The search stratagy is based on: 
Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollár, P. (2020). 
Designing Network Design Spaces. Retrieved from http://arxiv.org/abs/2003.13678

Shenda Hong, Apr 2020
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from util import read_data_physionet_2, read_data_physionet_4, preprocess_physionet, read_data_physionet_4_with_val
from net1d import Net1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary

def run_exp(base_filters, filter_mul_list, m_blocks_list):

    dataset = MyDataset(X_train, Y_train)
    dataset_val = MyDataset(X_test, Y_test)
    dataset_test = MyDataset(X_test, Y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)
    
    # make model
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model = Net1D(
        in_channels=1, 
        base_filters=base_filters, 
        ratio=1.0, 
        filter_mul_list=filter_mul_list, 
        m_blocks_list=m_blocks_list, 
        kernel_size=16, 
        stride=2, 
        groups_width=16,
        verbose=True, 
        n_classes=4)
    model.to(device)

    summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)
    exit()

    # train and test
    model.verbose = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()

    n_epoch = 50
    step = 0
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
                    
        # val
        model.eval()
        prog_iter_val = tqdm(dataloader_val, desc="Validation", leave=False)
        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_val):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                all_pred_prob.append(pred.cpu().data.numpy())
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)
        ## vote most common
        final_pred = []
        final_gt = []
        for i_pid in np.unique(pid_val):
            tmp_pred = all_pred[pid_val==i_pid]
            tmp_gt = Y_val[pid_val==i_pid]
            final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
            final_gt.append(Counter(tmp_gt).most_common(1)[0][0])
        ## classification report
        tmp_report = classification_report(final_gt, final_pred, output_dict=True)
        print(confusion_matrix(final_gt, final_pred))
        f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'] + tmp_report['2']['f1-score'] + tmp_report['3']['f1-score'])/4
        writer.add_scalar('F1/f1_score', f1_score, _)
        writer.add_scalar('F1/label_0', tmp_report['0']['f1-score'], _)
        writer.add_scalar('F1/label_1', tmp_report['1']['f1-score'], _)
        writer.add_scalar('F1/label_2', tmp_report['2']['f1-score'], _)
        writer.add_scalar('F1/label_3', tmp_report['3']['f1-score'], _)
                    
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
        ## vote most common
        final_pred = []
        final_gt = []
        for i_pid in np.unique(pid_test):
            tmp_pred = all_pred[pid_test==i_pid]
            tmp_gt = Y_test[pid_test==i_pid]
            final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
            final_gt.append(Counter(tmp_gt).most_common(1)[0][0])
        ## classification report
        tmp_report = classification_report(final_gt, final_pred, output_dict=True)
        print(confusion_matrix(final_gt, final_pred))
        f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'] + tmp_report['2']['f1-score'] + tmp_report['3']['f1-score'])/4
        writer.add_scalar('F1/f1_score', f1_score, _)
        writer.add_scalar('F1/label_0', tmp_report['0']['f1-score'], _)
        writer.add_scalar('F1/label_1', tmp_report['1']['f1-score'], _)
        writer.add_scalar('F1/label_2', tmp_report['2']['f1-score'], _)
        writer.add_scalar('F1/label_3', tmp_report['3']['f1-score'], _)


if __name__ == "__main__":

    batch_size = 32

    is_debug = True
    if is_debug:
        writer = SummaryWriter('/nethome/shong375/log/regnet/challenge2017/debug')
    else:
        writer = SummaryWriter('/nethome/shong375/log/regnet/challenge2017/run')

    # make data, (sample, channel, length)
    X_train, X_val, X_test, Y_train, Y_val, Y_test, pid_val, pid_test = read_data_physionet_4_with_val()
    print(X_train.shape, Y_train.shape)

    base_filters = 64
    w_a = 2.5
    filter_mul_list=[1,w_a,w_a,w_a**2,w_a**2,w_a**3,w_a**3]
    m_blocks_list=[2,2,2,3,3,4,4]

    run_exp(
        base_filters=base_filters,
        filter_mul_list=filter_mul_list,
        m_blocks_list=m_blocks_list)