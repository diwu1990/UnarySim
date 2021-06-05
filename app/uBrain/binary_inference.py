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
import shutil
import glob

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
# from torchsummaryX import summary
from torchinfo import summary
import matplotlib.pyplot as plt
import argparse

from binary_model import Cascade_CNN_RNN_Binary_test, Cascade_CNN_RNN_Binary

# parse input
parser = argparse.ArgumentParser()

hpstr = "set input dataset directory"
parser.add_argument('-idir', '--input_directory', default="/mnt/ssd1/data/bci/preprocessed_data/", type=str, help=hpstr)

hpstr = "set data scaling threshold"
parser.add_argument('-t', '--threshold', default=2., type=float, help=hpstr)

hpstr = "set output model directory"
parser.add_argument('-odir', '--output_directory', default="/home/diwu/Project/UnarySim/app/uBrain/saved_model/", type=str, help=hpstr)

hpstr = "set input size for a clip"
parser.add_argument('-i', '--input_sz', default=(10, 11), type=tuple, help=hpstr)

hpstr = "set activation function of linear layers"
parser.add_argument('-a', '--linear_act', default="scalerelu", type=str, help=hpstr)

hpstr = "set cnn channel size"
parser.add_argument('-c', '--cnn_chn', default=32, type=int, help=hpstr)

hpstr = "set cnn kernel size"
parser.add_argument('-k', '--cnn_kn_sz', default=3, type=int, help=hpstr)

hpstr = "set cnn padding size"
parser.add_argument('-p', '--cnn_padding', default=1, type=int, help=hpstr)

hpstr = "set fc size"
parser.add_argument('-f', '--fc_sz', default=1024, type=int, help=hpstr)

hpstr = "set rnn type"
parser.add_argument('-r', '--rnn', default="mgu", type=str, help=hpstr)

hpstr = "set rnn window size"
parser.add_argument('-w', '--rnn_win_sz', default=10, type=int, help=hpstr)

hpstr = "set rnn hidden size"
parser.add_argument('-hsz', '--rnn_hidden_sz', default=64, type=int, help=hpstr)

hpstr = "set whether to use hard rnn activation"
parser.add_argument('--rnn_hard', action='store_true', help=hpstr)

hpstr = "set whether to use bias for all matmul"
parser.add_argument('--bias', action='store_true', help=hpstr)

hpstr = "set keep prob for dropout"
parser.add_argument('-dp', '--keep_prob', default=0.5, type=float, help=hpstr)

hpstr = "set train batch size"
parser.add_argument('-bstrain', '--train_batch_sz', default=1024, type=int, help=hpstr)

hpstr = "set test batch size"
parser.add_argument('-bstest', '--test_batch_sz', default=2048, type=int, help=hpstr)

hpstr = "set epoch"
parser.add_argument('-e', '--epoch', default=300, type=int, help=hpstr)

hpstr = "set learning rate"
parser.add_argument('-lr', '--lr', default=1e-3, type=float, help=hpstr)

hpstr = "set std for initialization"
parser.add_argument('-std', '--init_std', default=None, type=float, help=hpstr)

hpstr = "set weight decay"
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float, help=hpstr)

hpstr = "set restart iteration"
parser.add_argument('-t0', '--t0_restart', default=50, type=int, help=hpstr)

hpstr = "set window overlap"
parser.add_argument('-ol', '--win_overlap', default=5, type=int, help=hpstr)

hpstr = "set whether store model"
parser.add_argument('--set_store', action='store_true', help=hpstr)

args = parser.parse_args()

dataset_dir = args.input_directory
threshold=args.threshold
model_dir = args.output_directory
input_sz=args.input_sz
linear_act=args.linear_act
cnn_chn=args.cnn_chn
cnn_kn_sz=args.cnn_kn_sz
cnn_padding=args.cnn_padding
fc_sz=args.fc_sz
rnn=args.rnn
rnn_win_sz=args.rnn_win_sz
rnn_hidden_sz=args.rnn_hidden_sz
rnn_hard=args.rnn_hard
bias=args.bias
init_std=args.init_std
keep_prob=args.keep_prob
bin_train_batch_sz=args.train_batch_sz
bin_test_batch_sz=args.test_batch_sz
training_epochs=args.epoch
lr=args.lr
weight_decay=args.weight_decay
t0=args.t0_restart
win_overlap=args.win_overlap
set_store=args.set_store

num_class=5
pin_memory=True
non_blocking=False

np.random.seed(33)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# device configuration
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("Using CUDA...")
else:
    print("Using CPU...")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# dataset configuration
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("******************** Dataset Configuration Start ********************")

with open(dataset_dir+"preprocessed_1_108_shuffle_dataset_3D_win_"+str(rnn_win_sz)+"_overlap_"+str(win_overlap)+".pkl", "rb") as fp:
    datasets = pickle.load(fp)
with open(dataset_dir+"preprocessed_1_108_shuffle_labels_3D_win_"+str(rnn_win_sz)+"_overlap_"+str(win_overlap)+".pkl", "rb") as fp:
    labels = pickle.load(fp)

outlier_ratio=(np.sum(datasets > threshold) + np.sum(datasets < -threshold))/len(datasets)/10/11/10
print("Data scaling threshold: %1.1f" % threshold)
print("\tResultant outlier ratio: %2.3f %%" % (outlier_ratio*100))

datasets = datasets.reshape(len(datasets), rnn_win_sz, input_sz[0], input_sz[1])
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)

split = np.random.rand(len(datasets)) < 0.75

train_x = datasets[split]
train_y = labels[split]

train_sample = len(train_x)
print("# of train samples:\t", train_sample)

train_x_tensor = torch.Tensor(train_x).clamp(-threshold, threshold)/threshold
train_y_tensor = torch.Tensor(train_y).clamp(-threshold, threshold)/threshold

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
print("# of test samples:\t", test_sample)

test_x_tensor = torch.Tensor(test_x).clamp(-threshold, threshold)/threshold
test_y_tensor = torch.Tensor(test_y).clamp(-threshold, threshold)/threshold

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
# model = Cascade_CNN_RNN_Binary(input_sz=input_sz, # size of each window
model = Cascade_CNN_RNN_Binary_test(input_sz=input_sz, # size of each window
                                linear_act=linear_act, # activation in CNN
                                cnn_chn=cnn_chn, # input channle in CNN
                                cnn_kn_sz=cnn_kn_sz, # kernel (height, width) in CNN
                                cnn_padding=cnn_padding, # kernel (height, width) in CNN
                                fc_sz=fc_sz, # fc size in CNN and MLP
                                rnn=rnn, # gru or mgu
                                rnn_win_sz=rnn_win_sz, # window size in RNN
                                rnn_hidden_sz=rnn_hidden_sz, # hidden size in RNN
                                rnn_hard=rnn_hard, # flag to apply HardGRUCell RNN
                                bias=bias, # bias of matrix mul
                                init_std=init_std, # std for initialization of weight
                                keep_prob=keep_prob, # prob for drop out after each FC
                                num_class=num_class) # output size
model.to(device)
summary(model, (1, rnn_win_sz, input_sz[0], input_sz[1]))
print("********************** Model Configuration End **********************\n")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# train
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
print("**************************** Test Start *****************************")

filename=str(rnn)+"_hidden_"+str(rnn_hidden_sz)+"_cnn_chn_"+str(cnn_chn)+"_pad_"+str(cnn_padding)+"_act_"+str(linear_act)+"_fc_"+str(fc_sz)+"_std_"+str(init_std)+"_ol_"+str(win_overlap)+"_t_"+str(threshold)+"_e_"+str(training_epochs)+"_t0_"+str(t0)+"_lr_"+str(lr)+"_decay_"+str(weight_decay)
print("Target filename prefix: " + filename)
path=model_dir+filename+"_acc_*.pth.tar"
file=glob.glob(path)[0]
print("Loading model state dict: ", file)
model.load_state_dict(torch.load(file)["state_dict"])
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        inputs, labels = data[0].to(device, non_blocking=non_blocking), data[1].to(device, non_blocking=non_blocking)
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        total += torch.argmax(labels, dim=1).size(0)
        correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

    acc = 100 * correct / total
print("Test Accuracy: %3.3f %%" % (acc))

print("***************************** Test End ******************************\n")


