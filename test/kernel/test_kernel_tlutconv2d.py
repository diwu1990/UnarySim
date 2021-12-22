import torch
from UnarySim.kernel import TLUTConv2d
import matplotlib.pyplot as plt
import time
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_tlutconv2d():
    plot_en=False

    hwcfg={
        "temporal" : "w",
        "widtht" : 4,
        "formati" : "bfloat16",
        "widthi" : 12,
        "quantilei" : 1,
        "formatw" : "bfloat16",
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
    
    in_channels = 32
    out_channels = 16
    kernel_size = (3, 3)
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    bias = True
    padding_mode = 'zeros'

    total_bit = 8
    input_int_bit = 0
    input_fra_bit = total_bit - input_int_bit

    batch = 32
    input_size = (128, 32)
    input = ((torch.rand(batch, in_channels, input_size[0], input_size[1]) - 0.5) * 2).to(device).type(dtype)
    if hwcfg["formati"] == "fxp":
        input = torch.trunc(input << hwcfg["widthi"]).round() >> hwcfg["widthi"]
    input = input << input_int_bit

    conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, dtype=dtype).to(device)
    if hwcfg["formatw"] == "fxp":
        conv2d.weight.data = torch.trunc(conv2d.weight << hwcfg["widthw"]).round() >> hwcfg["widthw"]
        if bias:
            conv2d.bias.data = torch.trunc(conv2d.bias << hwcfg["widthw"]).round() >> hwcfg["widthw"]

    conv2d_o = conv2d(input)

    uconv2d = TLUTConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, 
                            weight_ext=conv2d.weight.data, bias_ext=conv2d.bias, hwcfg=hwcfg).to(device)
    uconv2d_o = uconv2d(input)
    print(uconv2d.hwcfg)

    conv2d_o.abs().mean().backward()
    uconv2d_o.abs().mean().backward()

    diff = (uconv2d_o - conv2d_o)
    print()
    print("diff max:", diff.max())
    print("diff min:", diff.min())
    print("diff mean:", diff.mean())
    print("diff rmse:", torch.sqrt(torch.mean(torch.square(diff))))

    diff_grad = (uconv2d.weight.grad - conv2d.weight.grad)
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
    test_tlutconv2d()

