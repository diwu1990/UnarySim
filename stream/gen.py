import torch
import numpy as np
from pylfsr import LFSR

def get_lfsr_seq(width=8):
    polylist = LFSR().get_fpolyList(m=width)
    poly = polylist[np.random.randint(0, len(polylist), 1)[0]]
    L = LFSR(fpoly=poly,initstate ='random')
    lfsr_seq = []
    for i in range(2**width):
        value = 0
        for j in range(width):
            value = value + L.state[j]*2**(width-1-j)
        lfsr_seq.append(value)
        L.next()
    return lfsr_seq


def get_sysrand_seq(width=8):
    return torch.randperm(2**width)
    
    
class RNG(torch.nn.Module):
    """
    Random number generator to return a random sequence of size [2**width] and type torch.nn.Parameter.
    """
    def __init__(
        self, 
        hwcfg={
            "width" : 8, 
            "dimr" : 1, 
            "rng" : "Sobol"
            },
        swcfg={
            "rtype" : torch.float
            }):
        super(RNG, self).__init__()
        self.hwcfg = {}
        self.hwcfg["width"] = hwcfg["width"]
        self.hwcfg["dimr"] = hwcfg["dimr"]
        self.hwcfg["rng"] = hwcfg["rng"].lower()

        self.swcfg = {}
        self.swcfg["rtype"] = swcfg["rtype"]

        self.width = hwcfg["width"]
        self.dimr = hwcfg["dimr"]
        self.rng = hwcfg["rng"].lower()
        self.seq_len = 2**self.width
        self.rng_seq = torch.nn.Parameter(torch.Tensor(1, self.seq_len), requires_grad=False)
        self.rtype = swcfg["rtype"]

        assert self.rng in ["sobol", "race", "lfsr", "sys", "rc", "tc"], \
            "Error: the hw config 'rng' in " + self + " class requires one of ['sobol', 'race', 'lfsr', 'sys', 'rc', 'tc']."
        if (self.rng == "sobol") or (self.rng == "rc"):
            # get the requested dimension of sobol random number
            self.rng_seq.data = torch.quasirandom.SobolEngine(self.dimr).draw(self.seq_len)[:, self.dimr-1].view(self.seq_len).mul_(self.seq_len)
        elif (self.rng == "race") or (self.rng == "tc"):
            self.rng_seq.data = torch.tensor([x/self.seq_len for x in range(self.seq_len)]).mul_(self.seq_len)
        elif self.rng == "lfsr":
            lfsr_seq = get_lfsr_seq(width=self.width)
            self.rng_seq.data = torch.tensor(lfsr_seq).type(torch.float)
        elif self.rng == "sys":
            sysrand_seq = get_sysrand_seq(width=self.width)
            self.rng_seq.data = sysrand_seq.type(torch.float)
        self.rng_seq.data = self.rng_seq.data.floor().type(self.rtype)

    def forward(self):
        return self.rng_seq


class RawScale(torch.nn.Module):
    """
    Scale raw data to [-1, 1], which meets bipolar/unipolar bitstream requirements.
    Data of type torch.nn.Parameter within (0, 100] percentile (%) of total data range are output.
    """
    def __init__(
        self, 
        hwcfg={
            "percentile" : 100
        }):
        super(RawScale, self).__init__()
        self.hwcfg = {}
        self.hwcfg["percentile"] = hwcfg["percentile"]

        self.percentile = hwcfg["percentile"]
        self.percentile_down = (100 - self.percentile) / 2 / 100
        self.percentile_up = (100 - self.percentile_down) / 100

    def forward(self, raw):
        lower_bound = torch.quantile(raw, self.percentile_down)
        upper_bound = torch.quantile(raw, self.percentile_up)
        scale = torch.max(lower_bound.abs(), upper_bound.abs())
        output = raw.clamp(lower_bound, upper_bound).div(scale)
        return output


class BinGen(torch.nn.Module):
    """
    Convert source data within [-1, 1] to binary integer data of type torch.nn.Parameter for comparison
    """
    def __init__(
        self, 
        source, 
        hwcfg={
            "width" : 8,
            "mode" : "bipolar"
        },
        swcfg={
            "rtype" : torch.float
        }):
        super(BinGen, self).__init__()
        self.hwcfg = {}
        self.hwcfg["width"] = hwcfg["width"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()

        self.swcfg = {}
        self.swcfg["rtype"] = swcfg["rtype"]

        self.source = source
        self.width = hwcfg["width"]
        self.mode = hwcfg["mode"].lower()
        self.rtype = swcfg["rtype"]
        assert self.mode in ["unipolar", "bipolar"], \
            "Error: the hw config 'mode' in " + self + " class requires one of ['unipolar', 'bipolar']."
        self.binary = torch.nn.Parameter(torch.Tensor(source.size()), requires_grad=False)
        if self.mode == "unipolar":
            self.binary.data = self.source
        elif self.mode == "bipolar":
            self.binary.data = self.source.add(1).div(2)
        self.binary.data = self.binary << self.width
        self.binary.data = self.binary.round().type(self.rtype)
        
    def forward(self):
        return self.binary


class BSGen(torch.nn.Module):
    """
    Compare binary data with rng[cycle] to generate bitstreams.
    "cycle" is used to indicate the generated bit at the current cycle.
    """
    def __init__(
        self, 
        binary, 
        rng, 
        swcfg={
            "stype" : torch.float
        }):
        super(BSGen, self).__init__()
        self.swcfg = {}
        self.swcfg["stype"] = swcfg["stype"]

        self.binary = binary
        self.rng = rng
        self.len = len(self.rng)
        self.stype = swcfg["stype"]
    
    def forward(self, cycle):
        return torch.gt(self.binary, self.rng[cycle.type(torch.long)%self.len]).type(self.stype)

