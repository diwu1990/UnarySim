#! /usr/bin/python3
import os
import numpy as np
import pandas as pd
import pyedflib

##############################################################################
# process raw edf data to csv format
##############################################################################

dataset_dir = "/home/dadafly/datasets/EEG_motor_imagery/"

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

###############################################################################
# convert all .event file to .csv file
###############################################################################
def convert_event_csv(dataset_dir):
	datadir_list=os.listdir(dataset_dir)
	for j in range(len(datadir_list)):
		if(os.path.isdir(dataset_dir+datadir_list[j])):
			print("***"+datadir_list[j]+"begin:")
			datadir_name = dataset_dir+datadir_list[j].strip()
			datafile_list = os.popen("ls "+datadir_name+"/*.edf").readlines()
			os.chdir(datadir_name)
			for k in range(len(datafile_list)):
				file_name = os.path.split(datafile_list[k])[1].strip()
				print(file_name+" begin:")
				file_name = os.path.split(datafile_list[k])[1].strip()
				csv_file_name = file_name.rstrip(".edf")+".event.csv"
				os.system("rdann -r "+file_name+" -f 0 -t 125 -a event -v >"+csv_file_name)
	return None
###############################################################################
# convert all .edf file to .csv file
###############################################################################
def convert_edf_csv(dataset_dir):
	datadir_list=os.listdir(dataset_dir)
	datadir_list=sorted(datadir_list)
	for j in range(len(datadir_list)):
		if(os.path.isdir(dataset_dir+datadir_list[j])):
			print("***"+datadir_list[j]+"begin:")
			datadir_name = dataset_dir+datadir_list[j].strip()
			datafile_list = os.popen("ls "+datadir_name+"/*.edf").readlines()
			os.chdir(datadir_name)
			for k in range(len(datafile_list)):
				file_name = os.path.split(datafile_list[k])[1].strip()
				print(file_name+" begin:")
				file_name = os.path.split(datafile_list[k])[1].strip()
				dataset_1D, label = read_data(datafile_list[k].strip())
				dataset_1D = pd.DataFrame(dataset_1D, columns=[list(label)])
				dataset_1D.to_csv(datafile_list[k].rstrip(".edf\n")+'.csv', index=False)
	return None
			
###################################################################################
# process label for all the data
###################################################################################
def gen_label(dataset_dir):
	datadir_list=os.listdir(dataset_dir)
	# handle each person
	for j in range(len(datadir_list)):
		if(os.path.isdir(data_set_dir+datadir_list[j])):
			datadir_name = dataset_dir+datadir_list[j].strip()
			datafile_list = os.popen("ls "+datadir_name+"/*.event.csv").readlines()
			os.chdir(datadir_name)
			# handle each task for one person
			for k in range(len(datafile_list)):
				file_name = os.path.split(datafile_list[k])[1].strip()
				print(file_name+" begin:")
				dataset_1D, label = read_data(datafile_list[k].strip())
				if("R01" in file_name):
					labels = pd.DataFrame(["eye_open"]*dataset_1D.shape[0], columns=["labels"])
					labels.to_csv(datafile_list[k].rstrip(".edf\n")+'.label.csv', index=False)
				elif("R02" in file_name):
					dataset_1D, label = read_data(datafile_list[k].strip())
					labels = pd.DataFrame(["eye_close"]*dataset_1D.shape[0], columns=["labels"])
					labels.to_csv(datafile_list[k].rstrip(".edf\n")+'.label.csv', index=False)
				elif(("R03" in file_name) or ("R07" in file_name) or ("R11" in file_name)):
					event = pd.read_fwf(datafile_list[k].strip())
					event["Aux"] = event["Aux"].astype(str)
					labels = []
					for p in range(len(event["Aux"])):
						# get the num of labels for one period
						if(p == len(event["Aux"])-1):
							label_num = dataset_1D.shape[0]-event["Sample #"][p]
						else:
							label_num = event["Sample #"][p+1]-event["Sample #"][p]
						# set labels for one period
						if("T0" in event["Aux"][p]):
							labels = labels+["rest"]*label_num
						elif("T1" in event["Aux"][p]):
							labels = labels+["open&close_left_fist"]*label_num
						elif("T2" in event["Aux"][p]):
							labels = labels+["open&close_right_fist"]*label_num
						else:
							print(event["Aux"][p])
					# output label file
					labels = pd.DataFrame(labels, columns=["labels"])
					labels.to_csv(datafile_list[k].rstrip(".edf\n")+'.label.csv', index=False)
				elif(("R04" in file_name) or ("R08" in file_name) or ("R12" in file_name)):
					event = pd.read_fwf(datafile_list[k].strip())
					event["Aux"] = event["Aux"].astype(str)
					labels = []
					for p in range(len(event["Aux"])):
						# get the num of labels for one period
						if(p == len(event["Aux"])-1):
							label_num = dataset_1D.shape[0]-event["Sample #"][p]
						else:
							label_num = event["Sample #"][p+1]-event["Sample #"][p]
						# set labels for one period
						if("T0" in event["Aux"][p]):
							labels = labels+["rest"]*label_num
						elif("T1" in event["Aux"][p]):
							labels = labels+["image_open&close_left_fist"]*label_num
						elif("T2" in event["Aux"][p]):
							labels = labels+["image_open&close_right_fist"]*label_num
						else:
							print(event["Aux"][p])
					# output label file
					labels = pd.DataFrame(labels, columns=["labels"])
					labels.to_csv(datafile_list[k].rstrip(".edf\n")+'.label.csv', index=False)
				elif(("R05" in file_name) or ("R09" in file_name) or ("R13" in file_name)):
					event = pd.read_fwf(datafile_list[k].strip())
					event["Aux"] = event["Aux"].astype(str)
					labels = []
					for p in range(len(event["Aux"])):
						# get the num of labels for one period
						if(p == len(event["Aux"])-1):
							label_num = dataset_1D.shape[0]-event["Sample #"][p]
						else:
							label_num = event["Sample #"][p+1]-event["Sample #"][p]
						# set labels for one period
						if("T0" in event["Aux"][p]):
							labels = labels+["rest"]*label_num
						elif("T1" in event["Aux"][p]):
							labels = labels+["open&close_both_fists"]*label_num
						elif("T2" in event["Aux"][p]):
							labels = labels+["open&close_both_feet"]*label_num
						else:
							print(event["Aux"][p])
					# output label file
					labels = pd.DataFrame(labels, columns=["labels"])
					labels.to_csv(datafile_list[k].rstrip(".edf\n")+'.label.csv', index=False)
				elif(("R06" in file_name) or ("R10" in file_name) or ("R14" in file_name)):
					event = pd.read_fwf(datafile_list[k].strip())
					event["Aux"] = event["Aux"].astype(str)
					labels = []
					for p in range(len(event["Aux"])):
						# get the num of labels for one period
						if(p == len(event["Aux"])-1):
							label_num = dataset_1D.shape[0]-event["Sample #"][p]
						else:
							label_num = event["Sample #"][p+1]-event["Sample #"][p]
						# set labels for one period
						if("T0" in event["Aux"][p]):
							labels = labels+["rest"]*label_num
						elif("T1" in event["Aux"][p]):
							labels = labels+["image_open&close_both_fists"]*label_num
						elif("T2" in event["Aux"][p]):
							labels = labels+["image_open&close_both_feet"]*label_num
						else:
							print(event["Aux"][p])
					# output label file
					labels = pd.DataFrame(labels, columns=["labels"])
					labels.to_csv(datafile_list[k].rstrip(".edf\n")+'.label.csv', index=False)
				else:
					print("*** ERROR ***")
				if(len(labels["labels"])==dataset_1D.shape[0]):
					print("yes")
				else:
					print(len(labels["labels"]))
					print(dataset_1D.shape[0])
					print("*** ERROR ***")
				print(file_name+" end\n")
		else:
			pass
	return None

if __name__ == '__main__':
	# convert_event_csv(dataset_dir)
	# gen_label(dataset_dir)
	convert_edf_csv(dataset_dir)
