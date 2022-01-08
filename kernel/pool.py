import torch
import math
from UnarySim.kernel import FSUAdd, NCFireStep, num2tuple

class FSUAvgPool2d(torch.nn.Module):
    """unary 2d average pooling based on scaled addition"""
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None,
                hwcfg={
                    "mode" : "bipolar", 
                    "depth" : 10
                }, 
                swcfg={
                    "btype" : torch.float, 
                    "stype" : torch.float
                }):
        super(FSUAvgPool2d, self).__init__()
        self.hwcfg = {}
        self.hwcfg["mode"] = hwcfg["mode"]
        self.hwcfg["depth"] = hwcfg["depth"]
        self.hwcfg["dima"] = 0
        self.hwcfg["entry"] = math.prod(num2tuple(kernel_size))
        self.hwcfg["scale"] = math.prod(num2tuple(kernel_size))
        
        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["stype"] = swcfg["stype"]

        # define the kernel avgpool2d
        self.pool = torch.nn.AvgPool2d(kernel_size, 
                                        stride=stride, 
                                        padding=padding, 
                                        ceil_mode=ceil_mode, 
                                        count_include_pad=count_include_pad, 
                                        divisor_override=1)

        # define teh kernel FSUAdd for scaled addition
        self.sadd = FSUAdd(self.hwcfg, self.swcfg)
        
    def forward(self, input):
        ws = self.pool(input.type(torch.float)).unsqueeze(0)
        output = self.sadd(ws).type(input.type())
        return output


class FSUAvgPool2dStoNe(torch.nn.Module):
    """
    2d average pooling for StoNe
    """
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None,
                hwcfg={
                    "mode" : "bipolar",
                    "scale" : None,
                    "leak" : 0.5,
                    "widthg" : 0.1
                }):
        super(FSUAvgPool2dStoNe, self).__init__()
        self.hwcfg = {}
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["scale"] = hwcfg["scale"]
        self.hwcfg["leak"] = hwcfg["leak"]
        self.hwcfg["widthg"] = hwcfg["widthg"]

        self.mode = hwcfg["mode"].lower()
        assert self.mode in ["unipolar", "bipolar"], \
            "Error: the hw config 'mode' in " + str(self) + " class requires one of ['unipolar', 'bipolar']."

        if self.hwcfg["scale"] is None:
            self.hwcfg["scale"] = math.prod(num2tuple(kernel_size))

        # define the kernel avgpool2d
        # no scaling should be done here
        self.pool = torch.nn.AvgPool2d(kernel_size, 
                                        stride=stride, 
                                        padding=padding, 
                                        ceil_mode=ceil_mode, 
                                        count_include_pad=count_include_pad, 
                                        divisor_override=1)

        self.leak_alpha = self.hwcfg["leak"]
        self.vth = self.hwcfg["scale"]
        if self.hwcfg["mode"] == "unipolar":
            self.m = 1.
            self.k = 0.
        else:
            self.m = 2.
            self.k = 1.

    def forward_bptt(self, input, u_prev):
        ws = self.pool(input.type(torch.float)*self.m-self.k)
        us = self.leak_alpha * u_prev + ws.type(input.type())
        out = NCFireStep.apply(us, self.vth, self.hwcfg["widthg"]).type(input.type())
        u = us - self.vth * (out*self.m-self.k)
        return out, us, u

    def forward(self, input, u_prev):
        return self.forward_bptt(input, u_prev)

