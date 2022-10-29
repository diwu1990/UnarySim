import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# extract accuracy from logs

log_dir = "/home/diwu/Project/UnarySim/app/uBrain/model_eval/sp/"
bitwidth_list = [4, 5, 6, 7, 8, 9, 10, 11, 12]
flag = "Test Accuracy:"


yellow = "#FFCA68"
orange = "#FFA668"
green = "#71DBBA"
gray = '#919191'
blue = "#6D9CF5"
red = "#FD661F"
brown = "#9E7446"
cactus = "#9CC424"


# extract fp log
log_file = log_dir + "log_fp.log"
fp = open(log_file, "r")
for line in fp:
    line = line.rstrip()  # remove '\n' at end of line
    if flag in line:
        line_list = line.split()
        acc_fp = float(line_list[2])
        print("FP accuracy: ", acc_fp, "%")
        break
fp.close()


# extract fxp log
acc_fxp = []
for bitwidth in bitwidth_list:
    log_file = log_dir + "log_fxp_"+str(bitwidth)+".log"
    fp = open(log_file, "r")
    for line in fp:
        line = line.rstrip()  # remove '\n' at end of line
        if flag in line:
            line_list = line.split()
            acc_fxp.append(float(line_list[2]))
            print(bitwidth, "-bit FXP accuracy: ", acc_fxp[-1], "%")
            break
    fp.close()


# extract sc log
acc_sc = []
for bitwidth in bitwidth_list:
    log_file = log_dir + "log_sc_bwrc_"+str(bitwidth+1)+"_bwtc_"+str(bitwidth)+".log"
    fp = open(log_file, "r")
    for line in fp:
        line = line.rstrip()  # remove '\n' at end of line
        if flag in line:
            line_list = line.split()
            acc_sc.append(float(line_list[2]))
            print(bitwidth, "-bit SC accuracy: ", acc_sc[-1], "%")
            break
    fp.close()


# extract hub log
acc_hub = []
for bitwidth in bitwidth_list:
    log_file = log_dir + "log_hub_"+str(bitwidth)+".log"
    fp = open(log_file, "r")
    for line in fp:
        line = line.rstrip()  # remove '\n' at end of line
        if flag in line:
            line_list = line.split()
            acc_hub.append(float(line_list[2]))
            print(bitwidth, "-bit HUB accuracy: ", acc_hub[-1], "%")
            break
    fp.close()


# extract hub noisy log
acc_hno = []
for bitwidth in bitwidth_list:
    log_file = log_dir + "log_hub_noisy_"+str(bitwidth)+".log"
    fp = open(log_file, "r")
    for line in fp:
        line = line.rstrip()  # remove '\n' at end of line
        if flag in line:
            line_list = line.split()
            acc_hno.append(float(line_list[2]))
            print(bitwidth, "-bit HNO accuracy: ", acc_hno[-1], "%")
            break
    fp.close()


font = {'family':'Times New Roman', 'size': 6}
matplotlib.rc('font', **font)

my_dpi = 300
fig_h = 1
fig_w = 3.6
alpha = 1

lw = 1
ms = 1.5

labels = [str(bitwidth) for bitwidth in bitwidth_list]
labels.append("FP32")
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=my_dpi)

x_axe = x[0:-1]
ax.plot(x[-1], acc_fp, "o", label="CPU", alpha=alpha, color=gray, lw=lw, ms=ms)
ax.plot(x_axe, acc_fxp, "-^", label="Systolic", alpha=alpha, color=blue, lw=lw, ms=ms)
# ax.plot(x_axe, acc_sc, "-P", label="SC", alpha=alpha, color=blue, lw=lw, ms=ms)
ax.plot(x_axe, acc_hub, "-s", label="uBrain", alpha=alpha, color=red, lw=lw, ms=ms)

locs = [50, 60, 70, 80, 90]
ax.set_yticks(locs)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('Data bitwidth')
ax.set_ylabel('Accuracy (%)')
ax.legend(loc='center right', ncol=2, frameon=True)
fig.tight_layout()
fig.savefig(log_dir+"model_eval_acc_sp_slides.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)
