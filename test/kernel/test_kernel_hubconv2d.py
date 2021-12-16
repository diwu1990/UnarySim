import torch
from UnarySim.kernel import HUBConv2d
import matplotlib.pyplot as plt
import time
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_hubconv2d():
    plot_en=False

    hwcfg={
        "widthi" : 12,
        "rngi" : "Sobol",
        "quantilei" : 1,
        "widthw" : 12,
        "rngw" : "Race",
        "quantilew" : 1,
        "cycle" : 2049,
        "rounding" : "round",
        "signmag" : True
    }
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

    input_size = (128, 32)
    input = ((torch.rand(32, in_channels, input_size[0], input_size[1]) - 0.5) * 2).to(device)
    input = input << input_int_bit

    conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode).to(device)
    conv2d_o = conv2d(input)

    uconv2d = HUBConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, 
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

    batch = 2
    inp = torch.randn(batch, 3, 10, 12)
    w_tensor = torch.randn(2, 3, 4, 5)

    fc_w = torch.nn.Parameter(w_tensor, requires_grad=True)
    fc_inp = torch.nn.functional.unfold(inp, (4, 5))
    print("input size: ", fc_inp.size())
    fc_inp = fc_inp.transpose(1, 2)
    print("input transpose size: ", fc_inp.size())
    fc_inp = fc_inp.reshape(-1, fc_inp.size()[-1])
    print("input reshape size: ", fc_inp.size())
    print()

    print("weight size: ", fc_w.size())
    fc_w = fc_w.view(fc_w.size()[0], -1)
    print("weight view size: ", fc_w.size())
    fc_w = fc_w.t()
    print("weight transpose size: ", fc_w.size())
    print()

    fc_out = fc_inp.matmul(fc_w)
    print("out size: ", fc_out.size())
    fc_out = fc_out.reshape(batch, -1, fc_out.size()[-1])
    print("out reshape size: ", fc_out.size())
    fc_out = fc_out.transpose(1, 2)
    print("out transpose size: ", fc_out.size())
    fc_out = torch.nn.functional.fold(fc_out, (7, 8), (1, 1))
    print("out fold size: ", fc_out.size())
    fc_out.sum().backward()
    print()

    conv_w = torch.nn.Parameter(w_tensor, requires_grad=True)
    conv_out = torch.nn.functional.conv2d(inp, conv_w)
    conv_out.sum().backward()
    (conv_out - fc_out).abs().max()


if __name__ == '__main__':
    test_hubconv2d()

