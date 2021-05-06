#! /usr/bin/python3

########################################################
# EEG data preprocess for 1D/2D/3D
########################################################
import argparse
import os
import pyedflib
import numpy as np
import pandas as pd
import pickle

np.random.seed(0)
def get_args():
	parser = argparse.ArgumentParser()
	
	hpstr = "set dataset directory"
	parser.add_argument('-d', '--directory', default="/home/dadafly/datasets/EEG_motor_imagery/", nargs='*', type=str, help=hpstr)

	hpstr = "set window size"
	parser.add_argument('-w', '--window', default=10, nargs='*', type=int, help=hpstr)

	hpstr = "set whether parallel"
	parser.add_argument('--parallel', action='store_true', help=hpstr)

	hpstr = "set whether convert to 2D matrix"
	parser.add_argument('--convert', action='store_true', help=hpstr)

	hpstr = "set whether segment data"
	parser.add_argument('--segment', action='store_true', help=hpstr)

	hpstr = "set begin person"
	parser.add_argument('-b', '--begin', default=1, nargs='?', type=int, help=hpstr)

	hpstr = "set end person"
	parser.add_argument('-e', '--end', default=108, nargs='?', type=int, help=hpstr)

	hpstr = "set output directory"
	parser.add_argument('-o', '--output_dir', default="/home/dadafly/datasets/EEG_motor_imagery/", nargs='*', help=hpstr)

	hpstr = "set whether store data"
	parser.add_argument('--set_store', action='store_true', help=hpstr)

	args = parser.parse_args()
	return(args)
		   
def print_top(dataset_dir, window_size, convert, segment, begin_subject, end_subject, output_dir, set_store):
	print("######################## PhysioBank EEG data preprocess ########################	\
		   \n#### Author: Dalin Zhang	UNSW, Sydney	email: zhangdalin90@gmail.com #####	\
		   \n# input directory:	%s \
		   \n# window size:		%d 	\
		   \n# parallel:	%s 	\
		   \n# convert:		%s 	\
		   \n# segment:		%s 	\
		   \n# begin subject:	%d 	\
		   \n# end subject:		%d 	\
		   \n# output directory:	%s	\
		   \n# set store:		%s 	\
		   \n##############################################################################"% \
			(dataset_dir,	\
			window_size,	\
			convert,		\
			segment,		\
			parallel,		\
			begin_subject,	\
			end_subject,	\
			output_dir,		\
			set_store))
	return None


def read_data(file_name):
	f = pyedflib.EdfReader(file_name)
	n = f.signals_in_file
	signal_labels = f.getSignalLabels()
	sigbufs = np.zeros((n, f.getNSamples()[0]))
	for i in np.arange(n):
	    sigbufs[i, :] = f.readSignal(i)
	sigbuf_transpose = np.transpose(sigbufs)
	signal = np.asarray(sigbuf_transpose)
	signal_labels = np.asarray(signal_labels)
	f._close()
	del f
	return signal, signal_labels

def data_1Dto2D(data, Y=10, X=11):
	data_2D = np.zeros([Y, X])
	data_2D[0] = ( 	   	 0, 	   0,  	   	 0, 	   0, data[21], data[22], data[23], 	   0,  	     0, 	   0, 	 	 0) 
	data_2D[1] = (	  	 0, 	   0,  	   	 0, data[24], data[25], data[26], data[27], data[28], 	   	 0,   	   0, 	 	 0) 
	data_2D[2] = (	  	 0, data[29], data[30], data[31], data[32], data[33], data[34], data[35], data[36], data[37], 	 	 0) 
	data_2D[3] = (	  	 0, data[38],  data[0],  data[1],  data[2],  data[3],  data[4],  data[5],  data[6], data[39], 		 0) 
	data_2D[4] = (data[42], data[40],  data[7],  data[8],  data[9], data[10], data[11], data[12], data[13], data[41], data[43]) 
	data_2D[5] = (	  	 0, data[44], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[45], 		 0) 
	data_2D[6] = (	  	 0, data[46], data[47], data[48], data[49], data[50], data[51], data[52], data[53], data[54], 		 0) 
	data_2D[7] = (	  	 0, 	   0, 	 	 0, data[55], data[56], data[57], data[58], data[59], 	   	 0, 	   0, 		 0) 
	data_2D[8] = (	  	 0, 	   0, 	 	 0, 	   0, data[60], data[61], data[62], 	   0, 	   	 0, 	   0, 		 0) 
	data_2D[9] = (	  	 0, 	   0, 	 	 0, 	   0, 	     0, data[63], 		 0, 	   0, 	   	 0, 	   0, 		 0) 
	return data_2D

def norm_dataset(dataset_1D):
	norm_dataset_1D = np.zeros([dataset_1D.shape[0], 64])
	for i in range(dataset_1D.shape[0]):
		norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
	return norm_dataset_1D

def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data.nonzero()].std()
	data_normalized = data
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
	return data_normalized

def dataset_1Dto2D(dataset_1D):
	dataset_2D = np.zeros([dataset_1D.shape[0], 10, 11])
	for i in range(dataset_1D.shape[0]):
		dataset_2D[i] = data_1Dto2D(dataset_1D[i])
	return dataset_2D

def norm_dataset_1Dto2D(dataset_1D):
	norm_dataset_2D = np.zeros([dataset_1D.shape[0], 10, 11])
	for i in range(dataset_1D.shape[0]):
		norm_dataset_2D[i] = feature_normalize(data_1Dto2D(dataset_1D[i]))
	return norm_dataset_2D

def windows(data, size):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		start += (size/2)

def segment_signal_without_transition(data, label, window_size):
	for (start, end) in windows(data, window_size):
		if((len(data[start:end]) == window_size) and (len(set(label[start:end]))==1)):
			if(start == 0):
				segments = data[start:end]
				# labels = stats.mode(label[start:end])[0][0]
				labels = np.array(list(set(label[start:end])))
			else:
				segments = np.vstack([segments, data[start:end]])
				labels = np.append(labels, np.array(list(set(label[start:end]))))
				# labels = np.append(labels, stats.mode(label[start:end])[0][0])
	return segments, labels

def apply_mixup(dataset_dir, parallel, convert, segment, window_size, start=1, end=110):
	# initial empty label arrays
	label_inter	= np.empty([0])
	# initial empty data arrays
	if (parallel == True):
		data_inter_cnn	= np.empty([0, window_size, 10, 11])
		data_inter_rnn	= np.empty([0, window_size, 64])
	elif ((convert == False) and (segment == False)):
		data_inter	= np.empty([0, 64])
	elif ((convert == False) and (segment == True)): 
		data_inter	= np.empty([0, window_size, 64])
	elif ((convert == True) and (segment == False)): 
		data_inter	= np.empty([0, 10, 11])
	elif ((convert == True) and (segment == True)): 
		data_inter	= np.empty([0, window_size, 10, 11])
	for j in range(start, end):
		if (j == 89):
			j = 109
		# get directory name for one subject
		data_dir = dataset_dir+"S"+format(j, '03d')
		# get task list for one subject
		task_list = [task for task in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, task))]
		for task in task_list:
			if(("R02" in task) or ("R04" in task) or ("R06" in task)): # R02: eye closed; R04, R06: motor imagery tasks
				print(task+" begin:")
				# get data file name and label file name
				data_file 	= data_dir+"/"+task+"/"+task+".csv"
				label_file 	= data_dir+"/"+task+"/"+task+".label.csv"
				# read data and label
				data		= pd.read_csv(data_file)
				label		= pd.read_csv(label_file)
				# remove rest label and data during motor imagery tasks
				data_label	= pd.concat([data, label], axis=1)
				data_label	= data_label.loc[data_label['labels']!= 'rest']
				# get new label
				label		= data_label['labels']
				# get new data and normalize
				data_label.drop('labels', axis=1, inplace=True)
				data		= data_label.as_matrix()
				data		= norm_dataset(data)
				if (parallel == True):
					# segment data
					data, label	= segment_signal_without_transition(data, label, window_size)
					# cnn data process
					data_cnn	= dataset_1Dto2D(data)
					data_cnn	= data_cnn.reshape(int(data_cnn.shape[0]/window_size), window_size, 10, 11)
					# rnn data process
					data_rnn	= data_cnn.reshape(int(data.shape[0]/window_size), window_size, 64)
				elif ((convert == False) and (segment == False)):
					pass
				elif ((convert == False) and (segment == True)):
					# segment data with sliding window 
					data, label	= segment_signal_without_transition(data, label, window_size)
					data		= data.reshape(int(data.shape[0]/window_size), window_size, 64)
				elif ((convert == True) and (segment == False)): 
					# convert 1D data to 2D
					data		= dataset_1Dto2D(data)
				elif ((convert == True) and (segment == True)): 
					# convert 1D data to 2D
					data		= dataset_1Dto2D(data)
					# segment data with sliding window 
					data, label	= segment_signal_without_transition(data, label, window_size)
					data		= data.reshape(int(data.shape[0]/window_size), window_size, 10, 11)
				# append new data and label
				if (parallel == True):
					data_inter_cnn	= np.vstack([data_inter_cnn, data_cnn])
					data_inter_rnn	= np.vstack([data_inter_rnn, data_rnn])
					label_inter	= np.append(label_inter, label)
				else:
					data_inter	= np.vstack([data_inter, data])
					label_inter	= np.append(label_inter, label)
			else:
				pass
	# shuffle data
	index = np.array(range(0, len(label_inter)))
	np.random.shuffle(index)
	if (parallel==True):
		shuffled_data_cnn	= data_inter_cnn[index]
		shuffled_data_rnn	= data_inter_rnn[index]
		shuffled_label 	= label_inter[index]
	else:
		shuffled_data	= data_inter[index]
		shuffled_label 	= label_inter[index]
	return shuffled_data, shuffled_label

if __name__ == '__main__':
	dataset_dir		=	get_args().directory
	window_size		=	get_args().window
	parallel		=	get_args().parallel
	convert			=	get_args().convert
	segment			=	get_args().segment
	begin_subject	=	get_args().begin
	end_subject		=	get_args().end
	output_dir		=	get_args().output_dir
	set_store		=	get_args().set_store
	print_top(dataset_dir, window_size, parallel, convert, segment, begin_subject, end_subject, output_dir, set_store)

	shuffled_data, shuffled_label = apply_mixup(dataset_dir, parallel, convert, segment, window_size, begin_subject, end_subject+1)
	if (set_store == True):
		if (parallel == True):
			output_data_cnn = output_dir+"parallel_cnn_rnn/raw_data/"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_cnn_dataset.pkl"
			output_data_rnn = output_dir+"parallel_cnn_rnn/raw_data/"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_rnn_dataset.pkl"
			output_label= output_dir+"parallel_cnn_rnn/raw_data/"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_labels.pkl"
		elif ((convert == False) and (segment == False)):
			output_data = output_dir+"1D_CNN/raw_data/"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_dataset_1D.pkl"
			output_label= output_dir+"1D_CNN/raw_data/"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_labels_1D.pkl"
		elif ((convert == False) and (segment == True)): 
			output_data = output_dir+"1D_CNN/raw_data/window_1D/"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_dataset_1D_win_"+str(window_size)+".pkl"
			output_label= output_dir+"1D_CNN/raw_data/window_1D/"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_labels_1D_win_"+str(window_size)+".pkl"
		elif ((convert == True) and (segment == False)): 
			output_data = output_dir+"2D_CNN/raw_data/"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_dataset_2D.pkl"
			output_label= output_dir+"2D_CNN/raw_data/"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_labels_2D.pkl"
		elif ((convert == True) and (segment == True)): 
			output_data = output_dir+"3D_CNN/raw_data/"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_dataset_3D_win_"+str(window_size)+".pkl"
			output_label= output_dir+"3D_CNN/raw_data/"+str(begin_subject)+"_"+str(end_subject)+"_shuffle_labels_3D_win_"+str(window_size)+".pkl"

		if (parallel ==True):
			with open(output_data_cnn, "wb") as fp:
				pickle.dump(shuffled_data_cnn, fp, protocol=4) 
			with open(output_data_rnn, "wb") as fp:
				pickle.dump(shuffled_data_rnn, fp, protocol=4) 
			with open(output_label, "wb") as fp:
				pickle.dump(shuffled_label, fp)
		else:
			with open(output_data, "wb") as fp:
				pickle.dump(shuffled_data, fp, protocol=4) 
			with open(output_label, "wb") as fp:
				pickle.dump(shuffled_label, fp)
