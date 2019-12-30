"""
train and test resnet1d on synthetic data, using ray

Usage: 
    (1) Install Ray:
        pip install ray --user
    (2) Run test on synthetic data
        python test_ray.py

    for the usage of Ray for PyTorch, please refer to: 
    https://ray.readthedocs.io/en/latest/using-ray-with-pytorch.html

Shenda Hong, Dec 2019
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

import ray

def train(model, device, train_loader, optimizer):
    
    loss_func = torch.nn.CrossEntropyLoss()
    all_loss = []
    prog_iter = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(prog_iter):

        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)

        loss = loss_func(pred, input_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_loss.append(loss.item())
    
def test(model, device, test_loader, label_test):

    prog_iter_test = tqdm(test_loader, desc="Testing", leave=False)
    all_pred_prob = []
    for batch_idx, batch in enumerate(prog_iter_test):
        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)
        all_pred_prob.append(pred.cpu().data.numpy())
    all_pred_prob = np.concatenate(all_pred_prob)
    all_pred = np.argmax(all_pred_prob, axis=1)
    ## classification report
    print(classification_report(all_pred, label_test))

class Network(object):
    def __init__(self, n_length, base_filters, kernel_size, n_block):
        """
        key parameters to control the model:
            n_length: dimention of input (resolution) [16, 64, 256, 1024, 4096]
            base_filters: number of convolutional filters (width) [8, 16, 32, 64, 128]
            kernel_size: size of convolutional filters [2, 4, 8, 16]
            n_block: depth of model (depth) [2, 4, 8, 16]
        """
        use_cuda = torch.cuda.is_available()

        n_samples = 1000
        n_length = n_length
        n_channel = 1
        n_classes = 2
        batch_size = 64

        data, label = read_data_generated(n_samples=n_samples, n_length=n_length, n_channel=n_channel, n_classes=n_classes)
        print(data.shape, Counter(label))
        dataset = MyDataset(data, label)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        data_test, label_test = read_data_generated(n_samples=n_samples, n_length=n_length, n_channel=n_channel, n_classes=n_classes)
        self.label_test = label_test
        print(data_test.shape, Counter(label_test))
        dataset_test = MyDataset(data_test, label_test)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)
        
        self.device = device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = dataloader, dataloader_test

        ## change the hyper-parameters for your own data
        # recommend: (n_block, downsample_gap, increasefilter_gap) = (8, 1, 2)
        # 34 layer (16*2+2): 16, 2, 4
        # 98 layer (48*2+2): 48, 6, 12
        self.model = ResNet1D(
                    in_channels=n_channel, 
                    base_filters=base_filters, 
                    kernel_size=kernel_size, 
                    stride=2, 
                    n_block=n_block, 
                    groups=base_filters,
                    n_classes=n_classes, 
                    downsample_gap=max(n_block//8, 1), 
                    increasefilter_gap=max(n_block//4, 1), 
                    verbose=False).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        train(self.model, self.device, self.train_loader, self.optimizer)
        return test(self.model, self.device, self.test_loader, self.label_test)

    def test(self):
        return test(self.model, self.device, self.test_loader, self.label_test)

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def save(self):
        torch.save(self.model.state_dict(), "synthetic_ray.pt")

    def load(self):
        self.model.load_state_dict(torch.load("synthetic_ray.pt"))

if __name__ == "__main__":

    # ------------------ test make model ------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for n_length in [16, 64, 256, 1024, 4096]:
        data, label = read_data_generated(n_samples=1000, n_length=n_length, n_channel=1, n_classes=2)
        for base_filters in [8, 16, 32, 64, 128]:
            for kernel_size in [2, 4, 8, 16]:
                for n_block in [2, 4, 8, 16]:
                    print('\n'*3)
                    print('*'*60)
                    print('*'*60)
                    print("n_length: {}, base_filters: {}, kernel_size: {}, n_block: {}".format(n_length, base_filters, kernel_size, n_block))
                    print('*'*60)
                    print('*'*60)
                    net = Network(
                        n_length=n_length, 
                        base_filters=base_filters, 
                        kernel_size=kernel_size, 
                        n_block=n_block)
                    net.model.to(device)
                    # summary(net.model, (data.shape[1], data.shape[2]))
                    net.test()

                    exit()

    # ------------------ test ray ------------------
    # ray.init()
    # RemoteNetwork = ray.remote(num_gpus=2)(Network)

    # NetworkActor = RemoteNetwork.remote()
    # NetworkActor2 = RemoteNetwork.remote()
    # ray.get([NetworkActor.train.remote(), NetworkActor2.train.remote()])

    # weights = ray.get(
    #     [NetworkActor.get_weights.remote(),
    #     NetworkActor2.get_weights.remote()])

    # from collections import OrderedDict
    # averaged_weights = OrderedDict(
    #     [(k, (weights[0][k] + weights[1][k]) / 2) for k in weights[0]])

    # weight_id = ray.put(averaged_weights)
    # [
    #     actor.set_weights.remote(weight_id)
    #     for actor in [NetworkActor, NetworkActor2]
    # ]
    # ray.get([actor.train.remote() for actor in [NetworkActor, NetworkActor2]])



