import torch
from UnarySim.kernel import FSUSignAbs
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.metric import ProgError
import matplotlib.pyplot as plt
import time
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_fsusignabs():
    total_cnt = 5
    hwcfg = {
            "width" : 8,
            "mode" : "bipolar",
            "dimr" : 1,
            "rng" : "sobol",
            "scale" : 1,
            "depth" : 5
        }
    swcfg = {
            "rtype" : torch.float,
            "stype" : torch.float,
            "btype" : torch.float
        }
    bitwidth = hwcfg["width"]
    mode = hwcfg["mode"]

    print("========================================================")
    print(mode)
    print("========================================================")
    # all input values are non-negative
    low_bound = 0
    if mode == "unipolar":
        up_bound = 2**bitwidth
    elif mode == "bipolar":
        low_bound = -2**(bitwidth-1)
        up_bound = 2**(bitwidth-1)

    input_list = []
    for input_val in range(low_bound, up_bound+1, 1):
        input_list.append(input_val)
    
    input = torch.tensor(input_list).type(torch.float).div(up_bound).to(device)
#     input = torch.tensor([-1/256]).type(torch.float).div(up_bound).to(device)

    output = torch.abs(input).to(device)
    
    result_pe_total = []
    for rand_idx in range(1, total_cnt+1):
        outputPE = ProgError(output, hwcfg).to(device)
    
        inputPE  = ProgError(input, hwcfg).to(device)
        inputSRC = BinGen(input, hwcfg, swcfg)().to(device)

        dut = FSUSignAbs(hwcfg, swcfg).to(device)
        
        inputRNG = RNG(hwcfg, swcfg)().to(device)
        inputBS = BSGen(inputSRC, inputRNG, hwcfg, swcfg).to(device)
        with torch.no_grad():
            start_time = time.time()
            for i in range(2**bitwidth):
                input_bs = inputBS(torch.tensor([i]))
                inputPE.Monitor(input_bs)

                _, output_bs = dut(input_bs)
                outputPE.Monitor(output_bs)
        
        # get the result for different rng
        result_pe = outputPE()[1].cpu().numpy()
        result_pe_total.append(result_pe)
    
    # get the result for different rng
    result_pe_total = np.array(result_pe_total)
    
    #######################################################################
    # check the error of all simulation
    #######################################################################
    print("RMSE:{:1.4}".format(np.sqrt(np.mean(result_pe_total**2))))
    print("MAE: {:1.4}".format(np.mean(np.abs(result_pe_total))))
    print("bias:{:1.4}".format(np.mean(result_pe_total)))
    print("max: {:1.4}".format(np.max(result_pe_total)))
    print("min: {:1.4}".format(np.min(result_pe_total)))

    #######################################################################
    # check the error according to input value
    #######################################################################
    # max_total = np.max(result_pe_total, axis=0)
    # min_total = np.min(result_pe_total, axis=0)
    # avg_total = np.mean(result_pe_total, axis=0)
    
    # axis_len = outputPE()[1].size()[0]
    # input_x_axis = []
    # for axis_index in range(axis_len):
    #     input_x_axis.append((axis_index/(axis_len-1)*(up_bound-low_bound)+low_bound)/up_bound)
    # fig, ax = plt.subplots()
    # ax.fill_between(input_x_axis, max_total, avg_total, facecolor="red", alpha=0.75)
    # ax.fill_between(input_x_axis, avg_total, min_total, facecolor="blue", alpha=0.75)
    # ax.plot(input_x_axis, avg_total, label='Avg error', color="black", linewidth=0.3)
    # plt.tight_layout()
    # plt.xlabel('Input value')
    # plt.ylabel('Output error')
    # plt.xticks(np.arange(-1.1, 1.1, step=0.5))
    # # ax.xaxis.set_ticklabels([])
    # plt.xlim(-1, 1)
    # plt.yticks(np.arange(-1.1, 1.1, step=0.2))
    # # ax.yaxis.set_ticklabels([])
    # plt.ylim(-1, 1)
    # plt.grid(b=True, which="both", axis="y", linestyle="--", color="grey", linewidth=0.3)
    # fig.set_size_inches(2, 2)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    test_fsusignabs()

