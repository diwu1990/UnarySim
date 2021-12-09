import torch
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.kernel import ShiftReg

class FSUMul(torch.nn.Module):
    """
    This module is for unary multiplication, supporting static/in-stream, unipolar/bipolar.
    If the multiplier is static, then need to input the pre-scaled in_1 to port in_1_prob.
    If the multiplier is in-stream, in_1_prob is ignored.
    Please refer to
    1) uGEMM: Unary Computing Architecture for GEMM Applications
    2) uGEMM: Unary Computing for GEMM Applications
    """
    def __init__(
        self,
        in_1_prob=None,
        hwcfg={
            "width" : 8,
            "mode" : "bipolar",
            "static" : False,
            "rng" : "Sobol",
            "dimr" : 1
        },
        swcfg={
            "rtype" : torch.float,
            "stype" : torch.float
        }):
        super(FSUMul, self).__init__()

        self.hwcfg={}
        self.hwcfg["width"] = hwcfg["width"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["static"] = hwcfg["static"]
        self.hwcfg["rng"] = hwcfg["rng"].lower()
        self.hwcfg["dimr"] = hwcfg["dimr"]

        self.swcfg={}
        self.swcfg["rtype"] = swcfg["rtype"]
        self.swcfg["stype"] = swcfg["stype"]

        self.entry = 2**hwcfg["width"]
        self.static = hwcfg["static"]
        self.stype = swcfg["stype"]
        self.rtype = swcfg["rtype"]
        
        self.mode = hwcfg["mode"].lower()
        assert self.mode in ["unipolar", "bipolar"], \
            "Error: the hw config 'mode' in " + str(self) + " class requires one of ['unipolar', 'bipolar']."

        # the random number generator used in computation
        self.rng = RNG(hwcfg, swcfg)()
        
        if self.static is True:
            # the probability of in_1 used in static computation
            self.in_1_prob = in_1_prob
            assert in_1_prob is not None, \
                "Error: the static multiplier requires in_1_prob in " + str(self) + " class."
            # directly create an unchange bitstream generator for static computation
            self.source_gen = BinGen(self.in_1_prob, hwcfg, swcfg)()
            self.bsg = BSGen(self.source_gen, self.rng, {"stype" : torch.int8})
            # rng_idx is used later as an enable signal, get update every cycled
            self.rng_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
            
            # Generate two seperate bitstream generators and two enable signals for bipolar mode
            if self.mode == "bipolar":
                self.bsg_inv = BSGen(self.source_gen, self.rng, {"stype" : torch.int8})
                self.rng_idx_inv = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
        else:
            # use a shift register to store the count of 1s in one bitstream to generate data
            sr_hwcfg={
                "entry" : self.entry
            }
            self.sr = ShiftReg(sr_hwcfg, swcfg)
            self.rng_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
            if self.mode == "bipolar":
                self.rng_idx_inv = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

    def FSUMul_forward(self, in_0, in_1=None):
        if self.static is True:
            # for input0 is 0.
            path = in_0.type(torch.int8) & self.bsg(self.rng_idx)
            # conditional update for rng index when input0 is 1. The update simulates enable signal of bs gen.
            self.rng_idx.data = self.rng_idx.add(in_0.type(torch.long))
            
            if self.mode == "unipolar":
                return path
            else:
                # for input0 is 0.
                path_inv = (1 - in_0.type(torch.int8)) & (1 - self.bsg_inv(self.rng_idx_inv))
                # conditional update for rng_idx_inv
                self.rng_idx_inv.data = self.rng_idx_inv.add(1 - in_0.type(torch.long))
                return path | path_inv
        else:
            _, source = self.sr(in_1)
            path = in_0.type(torch.int8) & torch.gt(source, self.rng[self.rng_idx]).type(torch.int8)
            self.rng_idx.data = self.rng_idx.add(in_0.type(torch.long)) % self.entry

            if self.mode == "unipolar":
                return path
            else:
                # for input0 is 0.
                path_inv = (1 - in_0.type(torch.int8)) & (1 - torch.gt(source, self.rng[self.rng_idx_inv]).type(torch.int8))
                # conditional update for rng_idx_inv
                self.rng_idx_inv.data = self.rng_idx_inv.add(1 - in_0.type(torch.long)) % self.entry
                return path | path_inv
            
    def forward(self, in_0, in_1=None):
        return self.FSUMul_forward(in_0, in_1).type(self.stype)

