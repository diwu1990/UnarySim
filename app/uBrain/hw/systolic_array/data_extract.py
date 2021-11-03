import os
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from numpy import mean, median

def data_extract(technode=""):
    font = {'family':'Times New Roman', 'size': 6}
    matplotlib.rc('font', **font)

    arch_list = ["eyeriss"]
    network_list = ["uBrain"]
    bit_list = ["8"]
    ram_list = ["ddr3_w__spm"]

    cycle_list = []
    area_list = []
    power_list = []
    energy_list = []

    print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")
    print("Processing " + arch_list[0] + ":")
    print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")
    
    # binary parallel
    computing = "bp"
    name = arch_list[0] + "_" + bit_list[0].zfill(2) + "b_" + computing + "_" + "001c_" + network_list[0] + "_" + ram_list[0]
    if not os.path.exists("./outputs/" + technode + "/" + name):
        raise ValueError("Folder ./outputs/" + technode + "/" + name + " does not exist.")
    
    path = "./outputs/" + technode + "/" + name + "/simHwOut/"
    cycle_list.append(return_indexed_elems(  input_csv=path + name + "_throughput_real.csv", index=2)) # total cycle
                    

def prune(input_list):
    l = []

    for e in input_list:
        e = e.strip() # remove the leading and trailing characters, here space
        if e != '' and e != ' ':
            l.append(e)

    return l


def return_indexed_elems(input_csv=None, index=None):
    l = []

    csv_file = open(input_csv, "r")
    
    first = True
    for entry in csv_file:
        if first == True:
            first = False
            continue

        elems = entry.strip().split(',')
        elems = prune(elems)
        if len(elems) > 0:
            l.append(float(elems[index]))
    
    csv_file.close()

    return l


if __name__ == '__main__':
    data_extract()
