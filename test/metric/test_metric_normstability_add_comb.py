# %%
import torch
from UnarySim.metric.metric import NormStability, NSbuilder, Stability, ProgError
from UnarySim.stream.gen import RNG, SourceGen, BSGen
from UnarySim.kernel.add import GainesAdd
import random
import matplotlib.pyplot as plt

from matplotlib import ticker, cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time
import math
import numpy as np

from tqdm import tqdm

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# %%
def test(
    rng="Sobol", 
    total_cnt=100, 
    mode="unipolar", 
    bitwidth=8, 
    threshold=0.05
):
    ns_val=[0.25, 0.5, 0.75]
    stype = torch.float
    rtype = torch.float
    
    pbar = tqdm(total=3*total_cnt*(2**bitwidth))
    if mode is "unipolar":
        # all values in unipolar are non-negative
        low_bound = 0
        up_bound = 2**bitwidth
    elif mode is "bipolar":
        # values in bipolar are arbitrarily positive or negative
        low_bound = -2**(bitwidth-1)
        up_bound = 2**(bitwidth-1)

    input = [[],[]]
    for dim_y in range(up_bound, low_bound-1, -1):
        input[0].append([])
        input[1].append([])
        for dim_x in range(low_bound, up_bound+1, 1):
            input[0][up_bound-dim_y].append(dim_y)
            input[1][up_bound-dim_y].append(dim_x)

    input = torch.tensor(input, dtype=torch.float).div(up_bound).to(device)
    acc_dim = 0
    output = torch.mean(input, acc_dim).to(device)

    for ns in ns_val:
        print("# # # # # # # # # # # # # # # # # #")
        print("Target normstab:", ns)
        print("# # # # # # # # # # # # # # # # # #")

        result_ns_total = []
        input_ns_total = []
        output_ns_total = []

        for rand_idx in range(1, total_cnt+1):
            outputNS = NormStability(output, mode=mode, threshold=threshold).to(device)

            inputNS = NormStability(input, mode=mode, threshold=threshold).to(device)

            dut = GainesAdd(mode=mode, 
                                scaled=True, 
                                acc_dim=acc_dim, 
                                rng="Sobol", 
                                rng_dim=4, 
                                rng_width=1).to(device)

            inputBSGen = NSbuilder(bitwidth=bitwidth, 
                                   mode=mode, 
                                   normstability=ns, 
                                   threshold=threshold, 
                                   value=input, 
                                   rng_dim=rand_idx).to(device)

            start_time = time.time()
            with torch.no_grad():
                for i in range(2**bitwidth):
                    input_bs = inputBSGen()
                    inputNS.Monitor(input_bs)

                    output_bs = dut(input_bs)
                    outputNS.Monitor(output_bs)
                    pbar.update(1)

            # get the result for different rng
            input_ns = inputNS()
            output_ns = outputNS()

            result_ns = (output_ns/torch.mean(input_ns, acc_dim)).clamp(0, 1).cpu().numpy()
            result_ns_total.append(result_ns)
            input_ns_total.append(torch.mean(input_ns, acc_dim).cpu().numpy())
            output_ns_total.append(output_ns.cpu().numpy())
            # print("--- %s seconds ---" % (time.time() - start_time))

        # get the result for different rng
        result_ns_total = np.array(result_ns_total)
        input_ns_total = np.array(input_ns_total)
        output_ns_total = np.array(output_ns_total)
        #######################################################################
        # check the error of all simulation
        #######################################################################
        print("avg I NS:{:1.4}".format(np.mean(input_ns_total)))
        print("max I NS:{:1.4}".format(np.max(input_ns_total)))
        print("min I NS:{:1.4}".format(np.min(input_ns_total)))
        print()
        print("avg O NS:{:1.4}".format(np.mean(output_ns_total)))
        print("max O NS:{:1.4}".format(np.max(output_ns_total)))
        print("min O NS:{:1.4}".format(np.min(output_ns_total)))
        print()
        print("avg O/I NS:{:1.4}".format(np.mean(result_ns_total)))
        print("max O/I NS:{:1.4}".format(np.max(result_ns_total)))
        print("min O/I NS:{:1.4}".format(np.min(result_ns_total)))
        print()

        #######################################################################
        # check the error according to input value
        #######################################################################
        avg_total = np.mean(result_ns_total, axis=0)
        fig, ax = plt.subplots()
        fig.set_size_inches(5.5, 4)
        axis_len = outputNS()[1].size()[0]
        y_axis = []
        x_axis = []
        for axis_index in range(axis_len):
            y_axis.append((up_bound-axis_index/(axis_len-1)*(up_bound-low_bound))/up_bound)
            x_axis.append((axis_index/(axis_len-1)*(up_bound-low_bound)+low_bound)/up_bound)
        X, Y = np.meshgrid(x_axis, y_axis)
        Z = avg_total
        levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        cs = plt.contourf(X, Y, Z, levels, cmap=cm.RdBu, extend="both")
        cbar = fig.colorbar(cs)

        # plt.tight_layout()
        plt.xticks(np.arange(low_bound/up_bound, up_bound/up_bound+0.1, step=0.5))
        # ax.xaxis.set_ticklabels([])
        plt.yticks(np.arange(low_bound/up_bound, up_bound/up_bound+0.1, step=0.5))
        # ax.yaxis.set_ticklabels([])

        plt.show()
        plt.close()
    pbar.close()

# %%
test(rng="Sobol", total_cnt=100, mode="unipolar", bitwidth=8, threshold=0.05)

# %%
