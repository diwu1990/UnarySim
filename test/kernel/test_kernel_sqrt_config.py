# %%
import torch
from UnarySim.kernel.sqrt import FSUSqrt
from UnarySim.stream.gen import RNG, SourceGen, BSGen
from UnarySim.metric.metric import ProgError
import matplotlib.pyplot as plt
import time
import math
import numpy as np

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def test(rng="Sobol", 
         mode="unipolar", 
         bitwidth=8, 
         jk_trace=False, 
         emit=True, 
         total_cnt=100, 
         depth_kernel=1, 
         depth_emit=3, 
         depth_sync=2, 
         savepdf=False):
    stype = torch.float
    rtype = torch.float
    
    print("========================================================")
    print(mode)
    print("========================================================")
    # all input values are non-negative
    low_bound = 0
    if mode == "unipolar":
        up_bound = 2**bitwidth
    elif mode == "bipolar":
        low_bound = 0
        up_bound = 2**(bitwidth-1)

    input_list = []
    for input_val in range(low_bound, up_bound+1, 1):
        input_list.append(input_val)
    
    input = torch.tensor(input_list).type(torch.float).div(up_bound).to(device)

    output = torch.sqrt(input).to(device)
    outputPE = ProgError(output, mode=mode).to(device)
    
    inputPE  = ProgError(input,  mode=mode).to(device)
    inputSRC = SourceGen(input, bitwidth, mode=mode, rtype=rtype)().to(device)
    
    result_pe_total = []
    for rand_idx in range(1, total_cnt+1):
        dut_sqrt = FSUSqrt(mode=mode, 
                         jk_trace=jk_trace, 
                         depth_kernel=depth_kernel, 
                         rng="Sobol", 
                         rng_dim=4, 
                         emit=emit, 
                         depth_emit=depth_emit, 
                         depth_sync=depth_sync, 
                         stype=torch.float).to(device)
        
        inputRNG = RNG(bitwidth, rand_idx, rng, rtype)().to(device)
        inputBS = BSGen(inputSRC, inputRNG, stype).to(device)
        with torch.no_grad():
            start_time = time.time()
            for i in range(2**bitwidth):
                input_bs = inputBS(torch.tensor([i]))
                inputPE.Monitor(input_bs)

                ouyput_bs = dut_sqrt(input_bs)
                outputPE.Monitor(ouyput_bs)
        
        # get the result for different rng
        result_pe = outputPE()[1].cpu().numpy()
        result_pe = result_pe.flatten()
        result_pe_total.append(result_pe)
    
    # get the result for different rng
    result_pe_total = np.array(result_pe_total)
    
    #######################################################################
    # check the error of all simulation
    #######################################################################
    print("RMSE:{:1.4}".format(math.sqrt(np.mean(result_pe_total**2))))
    print("MAE: {:1.4}".format(np.mean(np.abs(result_pe_total))))
    print("bias:{:1.4}".format(np.mean(result_pe_total)))
    print("max: {:1.4}".format(np.max(result_pe_total)))
    print("min: {:1.4}".format(np.min(result_pe_total)))

    #######################################################################
    # check the error according to input value
    #######################################################################
    max_total = np.max(result_pe_total, axis=0)
    min_total = np.min(result_pe_total, axis=0)
    avg_total = np.mean(result_pe_total, axis=0)
    
    axis_len = outputPE()[1].size()[0]
    input_x_axis = []
    for axis_index in range(axis_len):
        input_x_axis.append((axis_index/(axis_len-1)*(up_bound-low_bound)+low_bound)/up_bound)
    fig, ax = plt.subplots()
    ax.fill_between(input_x_axis, max_total, avg_total, facecolor="red", alpha=0.75)
    ax.fill_between(input_x_axis, avg_total, min_total, facecolor="blue", alpha=0.75)
    ax.plot(input_x_axis, avg_total, label='Avg error', color="black", linewidth=0.3)
    plt.tight_layout()
    plt.xlabel('Input value')
    plt.ylabel('Output error')
    plt.xticks(np.arange(0, 1.1, step=0.5))
    # ax.xaxis.set_ticklabels([])
    plt.xlim(0, 1)
    plt.yticks(np.arange(-0.2, 0.4, step=0.2))
    # ax.yaxis.set_ticklabels([])
    plt.ylim(-0.3, 0.55)
    plt.grid(b=True, which="both", axis="y", linestyle="--", color="grey", linewidth=0.3)
    fig.set_size_inches(4, 2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if savepdf is True:
        if emit is True:
            plt.savefig(mode+"-bw"+str(bitwidth)+"-bit"+"-emitting"+"-s"+str(depth_sync)+"-k"+str(depth_kernel)+"-e"+str(depth_emit)+".pdf", 
                        dpi=300, 
                        bbox_inches='tight')
        else:
            if jk_trace is True:
                plt.savefig(mode+"-bw"+str(bitwidth)+"-bit"+"-inserting-JK"+".pdf", 
                            dpi=300, 
                            bbox_inches='tight')
            else:
                plt.savefig(mode+"-bw"+str(bitwidth)+"-bit"+"-inserting-IS"+"-k"+str(depth_kernel)+".pdf", 
                            dpi=300, 
                            bbox_inches='tight')
    plt.show()
    plt.close()

# %%
"""
# Following code is to test the best configuration for bit-emit sqrt
## The conclusion is that depth_kernel=1, depth_emit=3, depth_sync=2 is the best for different source rngs
"""

# %%
rng="Sobol"
mode="unipolar"
print("111")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=1, depth_sync=1)
print("121")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=2, depth_sync=1)
print("122")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=2, depth_sync=2)
print("132")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=3, depth_sync=2)
print("142")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=4, depth_sync=2)
print("133")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=3, depth_sync=3)
print("144")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=4, depth_sync=4)
print("211")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=1, depth_sync=1)
print("221")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=2, depth_sync=1)
print("222")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=2, depth_sync=2)
print("232")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=3, depth_sync=2)
print("242")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=4, depth_sync=2)
print("233")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=3, depth_sync=3)
print("244")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=4, depth_sync=4)
print("411")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=1, depth_sync=1)
print("421")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=2, depth_sync=1)
print("422")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=2, depth_sync=2)
print("432")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=3, depth_sync=2)
print("442")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=4, depth_sync=2)
print("433")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=3, depth_sync=3)
print("444")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=4, depth_sync=4)

# %%
rng="Sobol"
mode="bipolar"
print("111")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=1, depth_sync=1)
print("121")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=2, depth_sync=1)
print("122")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=2, depth_sync=2)
print("132")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=3, depth_sync=2)
print("142")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=4, depth_sync=2)
print("133")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=3, depth_sync=3)
print("144")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=1, depth_emit=4, depth_sync=4)
print("211")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=1, depth_sync=1)
print("221")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=2, depth_sync=1)
print("222")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=2, depth_sync=2)
print("232")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=3, depth_sync=2)
print("242")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=4, depth_sync=2)
print("233")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=3, depth_sync=3)
print("244")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=2, depth_emit=4, depth_sync=4)
print("411")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=1, depth_sync=1)
print("421")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=2, depth_sync=1)
print("422")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=2, depth_sync=2)
print("432")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=3, depth_sync=2)
print("442")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=4, depth_sync=2)
print("433")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=3, depth_sync=3)
print("444")
test(rng=rng, mode=mode, bitwidth=8, emit=True, jk_trace=False, total_cnt=10, depth_kernel=4, depth_emit=4, depth_sync=4)

# %%
