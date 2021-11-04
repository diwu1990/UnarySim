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
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from UnarySim.app.uBrain.model.model_fp import Cascade_CNN_RNN
from UnarySim.kernel.utils import tensor_unary_outlier

# parse input
parser = argparse.ArgumentParser()

hpstr = "set input dataset directory for motor imagery"
parser.add_argument('-idir_mi', '--input_directory_mi', default="/mnt/ssd1/data/bci/motor_imagery/", type=str, help=hpstr)

hpstr = "set input dataset directory for seizure prediction"
parser.add_argument('-idir_sp', '--input_directory_sp', default="/mnt/ssd1/data/bci/seizure_prediction/", type=str, help=hpstr)

hpstr = "set train for motor imagery"
parser.add_argument('--task_mi', action='store_true', help=hpstr)

hpstr = "set train for seizure prediction"
parser.add_argument('--task_sp', action='store_true', help=hpstr)

hpstr = "set data scaling threshold for motor imagery"
parser.add_argument('-tmi', '--threshold_mi', default=2., type=float, help=hpstr)

hpstr = "set data scaling threshold for seizure prediction"
parser.add_argument('-tsp', '--threshold_sp', default=2., type=float, help=hpstr)

hpstr = "set output model directory"
parser.add_argument('-odir', '--output_directory', default="/home/diwu/Project/UnarySim/app/uBrain/saved_model/", type=str, help=hpstr)

hpstr = "set input sample system for a clip:\n\t1) '10-10' for international 10-10 system;\n\t2) '10-20' for international 10-20 system"
parser.add_argument('-i', '--input_sample', default="10-10", type=str, help=hpstr)

hpstr = "set activation function of linear layers"
parser.add_argument('-a', '--linear_act', default="scalerelu", type=str, help=hpstr)

hpstr = "set cnn channel size"
parser.add_argument('-c', '--cnn_chn', default=16, type=int, help=hpstr)

hpstr = "set cnn kernel size"
parser.add_argument('-k', '--cnn_kn_sz', default=3, type=int, help=hpstr)

hpstr = "set cnn padding size"
parser.add_argument('-p', '--cnn_padding', default=1, type=int, help=hpstr)

hpstr = "set fc size"
parser.add_argument('-f', '--fc_sz', default=256, type=int, help=hpstr)

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
parser.add_argument('-wd', '--weight_decay', default=0.00005, type=float, help=hpstr)

hpstr = "set restart iteration"
parser.add_argument('-t0', '--t0_restart', default=50, type=int, help=hpstr)

hpstr = "set window overlap"
parser.add_argument('-ol', '--win_overlap', default=5, type=int, help=hpstr)

hpstr = "set whether store model"
parser.add_argument('--set_store', action='store_true', help=hpstr)

hpstr = "set train seed"
parser.add_argument('-s', '--seed', default=33, type=int, help=hpstr)

args = parser.parse_args()

dataset_dir_mi = args.input_directory_mi
dataset_dir_sp = args.input_directory_sp
if args.task_mi is True and args.task_sp is True:
    print("Train for both motor imagery and seizure prediction.\n")
    num_class=[5, 2]
elif args.task_mi is True and args.task_sp is False:
    print("Train for only motor imagery.\n")
    num_class=[5]
elif args.task_mi is False and args.task_sp is True:
    print("Train for only seizure prediction.\n")
    num_class=[2]
elif args.task_mi is False and args.task_sp is False:
    print("Train for no tasks, and end program.\n")
    quit()
else:
    quit()
threshold_mi=args.threshold_mi
threshold_sp=args.threshold_sp
model_dir = args.output_directory
if args.input_sample == "10-10":
    input_sz=[10, 11]
if args.input_sample == "10-20":
    input_sz=[5, 5]
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

pin_memory=True
non_blocking=False

np.random.seed(args.seed)


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
print("Apply international "+args.input_sample+" system.")
if args.input_sample == "10-10":
    system = "/preprocessed_data_10_10/"
else:
    system = "/preprocessed_data_10_20/"


# motor imagery
if args.task_mi is True:
    dataset_file_mi = dataset_dir_mi+system+\
        "preprocessed_1_108_shuffle_dataset_3D_win_"+str(rnn_win_sz)+"_overlap_"+str(win_overlap)+".pkl"
    label_file_mi = dataset_dir_mi+system+\
        "preprocessed_1_108_shuffle_labels_3D_win_"+str(rnn_win_sz)+"_overlap_"+str(win_overlap)+".pkl"
    print("dataset: "+dataset_file_mi)
    print("label: "+label_file_mi)
    with open(dataset_file_mi, "rb") as fp:
        datasets_mi = pickle.load(fp)
        datasets_mi = datasets_mi.reshape(len(datasets_mi), rnn_win_sz, input_sz[0], input_sz[1])
    with open(label_file_mi, "rb") as fp:
        # label_file_mi have labels in one-hot coding
        # labels_mi = np.asarray(pd.get_dummies(pickle.load(fp)), dtype = np.int8)
        labels_mi = pickle.load(fp)
    outlier_ratio=(np.sum(datasets_mi > threshold_mi) + np.sum(datasets_mi < -threshold_mi))/datasets_mi.size
    print("Data scaling threshold (motor imagery): %1.1f" % threshold_mi)
    print("\tResultant outlier ratio: %2.3f %%" % (outlier_ratio*100))
    fp.close()

if args.task_sp is True:
    dataset_file_sp = dataset_dir_sp+system+\
        "preprocessed_1_79_shuffle_dataset_3D_win_"+str(rnn_win_sz)+"_overlap_"+str(win_overlap)+".pkl"
    label_file_sp = dataset_dir_sp+system+\
        "preprocessed_1_79_shuffle_labels_3D_win_"+str(rnn_win_sz)+"_overlap_"+str(win_overlap)+".pkl"
    print("dataset: "+dataset_file_sp)
    print("label: "+label_file_sp)
    with open(dataset_file_sp, "rb") as fp:
        datasets_sp = pickle.load(fp)
        datasets_sp = datasets_sp.reshape(len(datasets_sp), rnn_win_sz, input_sz[0], input_sz[1])
    with open(label_file_sp, "rb") as fp:
        # label_file_sp have labels in one-hot coding
        # labels_sp = np.asarray(pd.get_dummies(pickle.load(fp)), dtype = np.int8)
        labels_sp = pickle.load(fp)
    outlier_ratio=(np.sum(datasets_sp > threshold_sp) + np.sum(datasets_sp < -threshold_sp))/datasets_sp.size
    print("Data scaling threshold (seizure prediction): %1.1f" % threshold_sp)
    print("\tResultant outlier ratio: %2.3f %%" % (outlier_ratio*100))
    fp.close()


if args.task_mi is True and args.task_sp is True:
    datasets_mi = datasets_mi.clip(-threshold_mi, threshold_mi)/threshold_mi
    datasets_sp = datasets_sp.clip(-threshold_sp, threshold_sp)/threshold_sp
    datasets = np.concatenate((datasets_mi, datasets_sp))
    # increase the label index of sp
    labels = np.asarray(pd.get_dummies(np.concatenate((labels_mi, labels_sp))), dtype = np.int8)
elif args.task_mi is True and args.task_sp is False:
    datasets = datasets_mi.clip(-threshold_mi, threshold_mi)/threshold_mi
    labels = np.asarray(pd.get_dummies(labels_mi), dtype = np.int8)
elif args.task_mi is False and args.task_sp is True:
    datasets = datasets_sp.clip(-threshold_sp, threshold_sp)/threshold_sp
    labels = np.asarray(pd.get_dummies(labels_sp), dtype = np.int8)


split = np.random.rand(len(datasets)) < 0.75

train_x = datasets[split]
train_y = labels[split]

train_sample = len(train_x)
print("# of train samples:\t", train_sample)

train_x_tensor = torch.Tensor(train_x)
train_y_tensor = torch.Tensor(train_y)

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

test_x_tensor = torch.Tensor(test_x)
test_y_tensor = torch.Tensor(test_y)

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
model = Cascade_CNN_RNN(input_sz=input_sz, # size of each window
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
print("**************************** Train Start ****************************")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, t0)

loss=0.0
best_acc=0.0
filename=str(rnn)+"_hidden_"+str(rnn_hidden_sz)+"_cnn_chn_"+str(cnn_chn)+"_pad_"+str(cnn_padding)+"_act_"+str(linear_act)+"_rnn_hard_"+str(rnn_hard)+"_fc_"+str(fc_sz)+"_std_"+str(init_std)+"_ol_"+str(win_overlap)+"_tmi_"+str(threshold_mi)+"_tsp_"+str(threshold_sp)+"_e_"+str(training_epochs)+"_t0_"+str(t0)+"_lr_"+str(lr)+"_decay_"+str(weight_decay)
if args.task_mi:
    filename=filename+"_task_mi"
if args.task_sp:
    filename=filename+"_task_sp"
print("Target file: "+filename+"\n")
iters = len(train_dataloader)

pbar = tqdm(range(training_epochs))
for epoch in pbar:
    model.train()
    total_time = time.time()
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device, non_blocking=non_blocking), data[1].to(device, non_blocking=non_blocking)
        optimizer.zero_grad() # zero the parameter gradients
        outputs = model(inputs) # forward
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward() # backward
        optimizer.step() # optimize
        scheduler.step(epoch + i / iters)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data[0].to(device, non_blocking=non_blocking), data[1].to(device, non_blocking=non_blocking)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total += torch.argmax(labels, dim=1).size()[0]
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        # remember best acc and save checkpoint
        acc = 100 * correct / total
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if set_store:
            torch.save(
                {
                'epoch': epoch + 1,
                'best_acc': best_acc,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                }, 
                model_dir+filename+'.check_point.tmp.pth.tar')
            if is_best:
                shutil.copyfile(model_dir+filename+'.check_point.tmp.pth.tar', model_dir+filename+'.model_best.tmp.pth.tar')
    
    pbar.set_description("Epoch %3d ==> LR: %1.7f; Train Loss: %3.3f; Test Accuracy: %3.3f %%" % (epoch, optimizer.param_groups[0]["lr"], loss.detach().cpu().item(), acc))

for weight in model.parameters():
    tensor_unary_outlier(weight, "weight")

if set_store:
    shutil.copyfile(model_dir+filename+'.model_best.tmp.pth.tar', model_dir+filename+'_acc_'+'%2.2f' % (best_acc)+'.pth.tar')
    os.remove(model_dir+filename+'.check_point.tmp.pth.tar')
    os.remove(model_dir+filename+'.model_best.tmp.pth.tar')

print("Total Epoch %3d:\tFinal Train Loss: %3.3f;\tBest Test Accuracy: %3.3f %%" % (training_epochs, loss.detach().cpu().item(), best_acc))
print("***************************** Train End *****************************\n")

