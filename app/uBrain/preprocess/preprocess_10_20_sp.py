#! /usr/bin/python3

########################################################
# EEG data preprocess for 3D
# This code is adapted from https://github.com/diwu1990/Cascade-Parallel/blob/master/data_preprocess/pre_process.py
# source dataset: https://www.nature.com/articles/sdata201939
########################################################
import argparse
import os
import numpy as np
import pandas as pd
import pickle
import glob
from tqdm import tqdm

np.random.seed(0)

def get_args():
	parser = argparse.ArgumentParser()

	hpstr = "set dataset directory"
	parser.add_argument('-d', '--directory', default="/mnt/ssd1/data/bci/seizure_prediction/neonatal_eeg_out/", nargs='*', type=str, help=hpstr)

	hpstr = "set window size"
	parser.add_argument('-w', '--window', default=10, nargs='*', type=int, help=hpstr)

	hpstr = "set window overlap size"
	parser.add_argument('-ol', '--overlap', default=5, nargs='*', type=int, help=hpstr)

	hpstr = "set begin person"
	parser.add_argument('-b', '--begin', default=1, nargs='?', type=int, help=hpstr)

	hpstr = "set end person"
	parser.add_argument('-e', '--end', default=24, nargs='?', type=int, help=hpstr)

	hpstr = "set output directory"
	parser.add_argument('-o', '--output_dir', default="/mnt/ssd1/data/bci/seizure_prediction/preprocessed_data_10_20", nargs='*', help=hpstr)

	hpstr = "set whether store data"
	parser.add_argument('--set_store', action='store_true', help=hpstr)

	args = parser.parse_args()
	return(args)


def print_top(dataset_dir, window_size, overlap_size, begin_subject, end_subject, output_dir, set_store):
	print(  "######################## PhysioBank EEG data preprocess ####################### \
			\n## Author: Jingjie Li, ECE, UW--Madison, WI, USA; Email: jingjie.li@wisc.edu ## \
			\n# input directory:    %s \
			\n# window size:        %d \
			\n# overlap size:       %d \
			\n# begin subject:      %d \
			\n# end subject:        %d \
			\n# output directory:   %s \
			\n# set store:          %s \
			\n###############################################################################"% \
			(dataset_dir,    \
			window_size,    \
			overlap_size,    \
			begin_subject,    \
			end_subject,    \
			output_dir,        \
			set_store))
	return None


def data_1Dto2D(data, Y=5, X=5):
	data_2D = np.zeros([Y, X])
	# data_2D[0] = (       0,        0,        0,        0, data[21], data[22], data[23],        0,        0,        0,        0)
	# data_2D[1] = (       0,        0,        0, data[24], data[25], data[26], data[27], data[28],        0,        0,        0)
	# data_2D[2] = (       0, data[29], data[30], data[31], data[32], data[33], data[34], data[35], data[36], data[37],        0)
	# data_2D[3] = (       0, data[38],  data[0],  data[1],  data[2],  data[3],  data[4],  data[5],  data[6], data[39],        0)
	# data_2D[4] = (data[42], data[40],  data[7],  data[8],  data[9], data[10], data[11], data[12], data[13], data[41], data[43])
	# data_2D[5] = (       0, data[44], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[45],        0)
	# data_2D[6] = (       0, data[46], data[47], data[48], data[49], data[50], data[51], data[52], data[53], data[54],        0)
	# data_2D[7] = (       0,        0,        0, data[55], data[56], data[57], data[58], data[59],         0,       0,        0)
	# data_2D[8] = (       0,        0,        0,        0, data[60], data[61], data[62],        0,         0,       0,        0)
	# data_2D[9] = (       0,        0,        0,        0,        0, data[63],        0,        0,         0,       0,        0)

	### JL: dummy data order test may not be used below
	data_2D[0] = (0.0, data[0], 0.0, data[1], 0.0)
	data_2D[1] = (data[4], data[2], data[6], data[3], data[5])
	data_2D[2] = (data[10], data[7], data[9], data[8], data[12])
	data_2D[3] = (data[11], data[14], data[16], data[15], data[13])
	data_2D[4] = (0.0, data[17], 0.0, data[18], 0.0)

	return data_2D


def norm_dataset(dataset_1D, num_channel = 18):
	norm_dataset_1D = np.zeros([dataset_1D.shape[0], num_channel])
	for i in range(dataset_1D.shape[0]):
		norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
	return norm_dataset_1D


def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data.nonzero()].std()
	data_normalized = data

	# JL: the following could report value warning
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma

	return data_normalized


def dataset_1Dto2D(dataset_1D, Y = 5, X = 5):
	dataset_2D = np.zeros([dataset_1D.shape[0], Y, X])
	for i in range(dataset_1D.shape[0]):
		dataset_2D[i] = data_1Dto2D(dataset_1D[i], Y, X)
	return dataset_2D


def windows(data, size, overlap):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		# each window overlaps adjacent ones by "overlap" samples
		start += overlap


def segment_signal_without_transition(data, label, window_size, overlap_size):
	for (start, end) in windows(data, window_size, overlap_size):
		if((len(data[start:end]) == window_size) and (len(set(label[start:end]))==1)):
			if(start == 0):
				segments    = data[start:end]
				# labels = stats.mode(label[start:end])[0][0]
				labels      = np.array(list(set(label[start:end])))
			else:
				segments    = np.vstack([segments, data[start:end]])
				labels      = np.append(labels, np.array(list(set(label[start:end]))))
				# labels = np.append(labels, stats.mode(label[start:end])[0][0])
	return segments, labels


def apply_mixup(dataset_dir, window_size, overlap_size, start=1, end=2):
	# initial empty label arrays
	label_inter     = np.empty([0])
	# initial empty data arrays
	data_inter      = np.empty([0, window_size, 5, 5])
	for j in tqdm(range(start, end)):
		# if (j == 89):
		#     j = 109
		# get directory name for one subject
		#data_dir = dataset_dir+"S"+format(j, '03d')

		#print(task_list)


		# get data file name and label file name
		data_file   = dataset_dir+"/"+"eeg"+str(j)+".csv"
		label_file  = dataset_dir+"/"+"eeg"+str(j)+".label.csv"
		# read data and label
		data        = pd.read_csv(data_file)
		## JL: to drop unneeded channels and sort channels by same order
		label       = pd.read_csv(label_file)
		# remove rest label and data during motor imagery tasks
		data_label  = pd.concat([data, label], axis=1)


		#data_label  = data_label.loc[data_label['labels']!= 'rest']
		# get new label
		label       = data_label['labels']
		# get new data and normalize
		data_label.drop('labels', axis=1, inplace=True)
		data        = data_label.to_numpy()
		data        = norm_dataset(data, 19)
		# convert 1D data to 2D
		data        = dataset_1Dto2D(data, Y = 5, X = 5)
		# segment data with sliding window
		print("complete 2d transform")
		print("data size: ", data.shape)
		data, label = segment_signal_without_transition(data, label, window_size, overlap_size)
		print("complete segment_signal_without_transition")
		data        = data.reshape(int(data.shape[0]/window_size), window_size, 5, 5)
		# append new data and label
		data_inter  = np.vstack([data_inter, data])
		label_inter = np.append(label_inter, label)
		print("complete task: ", j)

	# shuffle data
	index = np.array(range(0, len(label_inter)))
	np.random.shuffle(index)
	shuffled_data   = data_inter[index]
	shuffled_label  = label_inter[index]
	return shuffled_data, shuffled_label


if __name__ == '__main__':
	dataset_dir     =    get_args().directory
	window_size     =    get_args().window
	overlap_size    =    get_args().overlap
	begin_subject   =    get_args().begin
	end_subject     =    get_args().end
	output_dir      =    get_args().output_dir
	set_store       =    get_args().set_store
	if type(window_size) is list:
		window_size = window_size[0]
	if type(overlap_size) is list:
		overlap_size = overlap_size[0]
	if type(begin_subject) is list:
		begin_subject = begin_subject[0]
	if type(end_subject) is list:
		end_subject = end_subject[0]
	print_top(dataset_dir, window_size, overlap_size, begin_subject, end_subject, output_dir, set_store)

	shuffled_data, shuffled_label = apply_mixup(dataset_dir, window_size, overlap_size, begin_subject, end_subject+1)
	if (set_store == True):
		output_data = output_dir+"preprocessed_"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_dataset_3D_win_"+str(window_size)+"_overlap_"+str(overlap_size)+".pkl"
		output_label= output_dir+"preprocessed_"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_labels_3D_win_"+str(window_size)+"_overlap_"+str(overlap_size)+".pkl"

		print("Dumping data and label:\n")
		with open(output_data, "wb") as fp:
			pickle.dump(shuffled_data, fp, protocol=4)
			print("\tData dump complete!!!")
		with open(output_label, "wb") as fp:
			pickle.dump(shuffled_label, fp)
			print("\tLabel dump complete!!!")
