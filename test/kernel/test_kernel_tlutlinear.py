import torch
from UnarySim.kernel import TLUTLinear
import matplotlib.pyplot as plt
import time
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_tlutlinear():
    plot_en=False

    hwcfg={
        "temporal" : "w",
        "widtht" : 4,
        "formati" : "fxp",
        "widthi" : 12,
        "quantilei" : 1,
        "formatw" : "fxp",
        "widthw" : 12,
        "quantilew" : 1,
        "cycle" : None,
        "rounding" : "round",
        "signmag" : True
    }

    if hwcfg["formati"] == "bfloat16":
        dtype = torch.bfloat16
    elif hwcfg["formati"] == "float16":
        dtype = torch.float16
    elif hwcfg["formati"] == "float32":
        dtype = torch.float32
    else:
        if hwcfg["formatw"] == "bfloat16":
            dtype = torch.bfloat16
        elif hwcfg["formatw"] == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
    
    batch = 16
    in_feature = 256
    out_feature = 256
    bias = True

    input_int_bit = 3
    
    input = ((torch.rand(batch, in_feature) - 0.5) * 2).to(device).type(dtype)
    if hwcfg["formati"] == "fxp":
        input = torch.trunc(input << hwcfg["widthi"]).round() >> hwcfg["widthi"]
    input = input << input_int_bit

    fc = torch.nn.Linear(in_feature, out_feature, bias=bias, dtype=dtype).to(device)
    if hwcfg["formatw"] == "fxp":
        fc.weight.data = torch.trunc(fc.weight << hwcfg["widthw"]).round() >> hwcfg["widthw"]
        if bias:
            fc.bias.data = torch.trunc(fc.bias << hwcfg["widthw"]).round() >> hwcfg["widthw"]
    
    fc_o = fc(input)

    ufc = TLUTLinear(in_feature, out_feature, bias=bias, weight_ext=fc.weight, bias_ext=fc.bias, hwcfg=hwcfg).to(device)
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
    test_tlutlinear()

