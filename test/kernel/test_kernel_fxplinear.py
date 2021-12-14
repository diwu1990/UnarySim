import torch
from UnarySim.kernel import FXPLinear
import matplotlib.pyplot as plt
import time
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_fxplinear():
    plot_en=False

    hwcfg={
        "widthi" : 6,
        "quantilei" : 1,
        "widthw" : 8,
        "quantilew" : 1,
        "rounding" : "round"
    }

    batch = 16
    in_feature = 256
    out_feature = 256
    bias = False

    total_bit = 8
    input_int_bit = 0
    input_fra_bit = total_bit - input_int_bit
    
    input = ((torch.rand(batch, in_feature) - 0.5) * 2).to(device)
    input = input << input_int_bit

    fc = torch.nn.Linear(in_feature, out_feature, bias=bias).to(device)
    
    fc_o = fc(input)

    ufc = FXPLinear(in_feature, out_feature, bias=bias, weight_ext=fc.weight, bias_ext=fc.bias, hwcfg=hwcfg).to(device)
    ufc_o = ufc(input)
    print(ufc.hwcfg)

    fc_o.abs().mean().backward()
    ufc_o.abs().mean().backward()

    diff = (ufc_o - fc_o)
    print()
    print("diff max:", diff.max())
    print("diff min:", diff.min())
    print("diff mean:", diff.mean())
    print("diff rmse:", torch.sqrt(torch.mean(torch.square(diff))))

    diff_grad = (ufc.weight.grad - fc.weight.grad)
    print()
    print("diff grad max:", diff_grad.max())
    print("diff grad min:", diff_grad.min())
    print("diff grad mean:", diff_grad.mean())
    print("diff grad rmse:", torch.sqrt(torch.mean(torch.square(diff_grad))))

    if plot_en:
        fig = plt.hist(diff.cpu().detach().numpy().flatten(), bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram for output error")
        plt.show()

        fig = plt.hist(diff_grad.cpu().detach().numpy().flatten(), bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram for grad error")
        plt.show()


if __name__ == '__main__':
    test_fxplinear()

