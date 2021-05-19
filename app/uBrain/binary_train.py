# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Implementing a Cascade CNN RNN for EEG classification using PyTorch
# Author: Di Wu, ECE, UW--Madison, WI, USA; Email: di.wu@ece.wisc.edu
# mainly refering to:
# 1: URL: https://github.com/dalinzhang/Cascade-Parallel
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import os
import sys
import pandas as pd
import pickle
import numpy as np
import time

import math
import warnings
import numbers
from typing import List, Tuple, Optional, overload, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

from binary_model import Cascade_CNN_RNN_Binary


input_sz=(10, 11) # size of each window
linear_act="relu" # activation in CNN
cnn_chn=32 # channle in CNN
cnn_kn_sz=3 # kernel (height, width) in CNN
cnn_padding=0 # kernel (height, width) in CNN
fc_sz=1024 # fc size in CNN
rnn_win_sz=10 # window size in RNN
rnn_hidden_sz=1024 # hidden size in RNN
rnn_hard=False # flag to apply HardGRUCell RNN
keep_prob=0.5 # prob for drop out after each FC
num_class=5 # output size
bin_train_batch_sz=1024 # train batch size for binary model
bin_test_batch_sz=1024 # test batch size for binary model
training_epochs=300 # train epoch
pin_memory=True
non_blocking=True
lr=1e-4
enable_penalty=False
lambda_loss_amount=0.0005
np.random.seed(33)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# device configuration
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("Using CUDA...")



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# dataset configuration
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("******************** Dataset Configuration Start ********************")
dataset_dir = "/mnt/ssd1/data/bci/preprocessed_data/"

with open(dataset_dir+"preprocessed_1_108_shuffle_dataset_3D_win_10.pkl", "rb") as fp:
      datasets = pickle.load(fp)
with open(dataset_dir+"preprocessed_1_108_shuffle_labels_3D_win_10.pkl", "rb") as fp:
      labels = pickle.load(fp)

datasets = datasets.reshape(len(datasets), rnn_win_sz, input_sz[0], input_sz[1], 1)
one_hot_labels = np.array(list(pd.get_dummies(labels)))
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)

split = np.random.rand(len(datasets)) < 0.75

train_x = datasets[split]
train_y = labels[split]

train_sample = len(train_x)
print("# of train samples:", train_sample)
print(train_x.max())
print(train_x.min())
print(np.abs(train_x).mean())
print((np.sum(train_x > 1) + np.sum(train_x < -1))/train_sample/10/11/10)

train_x_tensor = torch.Tensor(train_x).squeeze(-1)
train_y_tensor = torch.argmax(torch.Tensor(train_y), dim=1)
print(train_x_tensor.shape)
print(train_y_tensor.shape)
train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
train_dataloader = DataLoader(
                                train_dataset, 
                                batch_size=bin_train_batch_sz, 
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=pin_memory, 
                                drop_last=True
                                )

test_x = datasets[~split]
test_y = labels[~split]

test_sample = len(test_x)
print("# of test samples:", test_sample)

test_x_tensor = torch.Tensor(test_x).squeeze(-1)
test_y_tensor = torch.argmax(torch.Tensor(test_y), dim=1)

test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
test_dataloader = DataLoader(
                                test_dataset, 
                                batch_size=bin_test_batch_sz, 
                                shuffle=False, 
                                num_workers=4,
                                pin_memory=pin_memory, 
                                drop_last=True
                                )

print("********************* Dataset Configuration End *********************\n")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# model configuration
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("********************* Model Configuration Start *********************")
model = Cascade_CNN_RNN_Binary(input_sz=input_sz, # size of each window
                                linear_act=linear_act, # activation in CNN
                                cnn_chn=cnn_chn, # input channle in CNN
                                cnn_kn_sz=cnn_kn_sz, # kernel (height, width) in CNN
                                cnn_padding=cnn_padding, # kernel (height, width) in CNN
                                fc_sz=fc_sz, # fc size in CNN and MLP
                                rnn_win_sz=rnn_win_sz, # window size in RNN
                                rnn_hidden_sz=rnn_hidden_sz, # hidden size in RNN
                                rnn_hard=rnn_hard, # flag to apply HardGRUCell RNN
                                keep_prob=keep_prob, # prob for drop out after each FC
                                num_class=num_class) # output size
model.to(device)
summary(model, (3, 1, rnn_win_sz, input_sz[0], input_sz[1]))
print("********************** Model Configuration End **********************\n")



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# train
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("**************************** Train Start ****************************")
criterion0 = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(training_epochs):
    model.train()
    total_time = time.time()
    load_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        start_time = time.time()
        inputs, labels = data[0].to(device, non_blocking=non_blocking), data[1].to(device, non_blocking=non_blocking)
        load_time += time.time() - start_time
        
        start_time = time.time()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        if enable_penalty is True:
            loss = criterion(outputs, labels) + lambda_loss_amount * criterion0(outputs, labels)
        else:
            loss = criterion(outputs, labels)
        forward_time += time.time() - start_time
        
        start_time = time.time()
        # backward
        loss.backward()
        # optimize
        optimizer.step()
        backward_time += time.time() - start_time
        
    print("\tTrain load: %.2f sec; forward: %.2f sec; backward: %.2f sec" %(load_time, forward_time, backward_time))
    
    load_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    model.eval()
#     model.apply(clipper)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            start_time = time.time()
            inputs, labels = data[0].to(device, non_blocking=non_blocking), data[1].to(device, non_blocking=non_blocking)
            load_time += time.time() - start_time
            
            start_time = time.time()
            outputs = model(inputs)
            forward_time += time.time() - start_time
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print("\tTest load: %.2f sec; forward: %.2f sec; backward: %.2f sec" %(load_time, forward_time, backward_time))

    print('Train - Epoch %d, Loss: %f, Test Accuracy: %f %%' \
          % (epoch, loss.detach().cpu().item(), 100 * correct / total))
    
    print("\tTotal: %.2f sec" %(time.time() - total_time))

print("***************************** Train End *****************************\n")