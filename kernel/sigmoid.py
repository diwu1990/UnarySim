import torch
from UnarySim.kernel import FSUAdd

class FSUHardsigmoid(torch.nn.Module):
    """
    This is a fsu scaled addition (x+1)/2.
    It works for both unipolar and bipolar bitstreams.
    """
    def __init__(
        self, 
        hwcfg={
            "mode" : "bipolar", 
            "scale" : 2,
            "dima" : 0,
            "depth" : 8,
            "entry" : 2
        }, 
        swcfg={
            "btype" : torch.float, 
            "stype" : torch.float
        }):
        super(FSUHardsigmoid, self).__init__()
        self.hwcfg = {}
        self.hwcfg["depth"] = hwcfg["depth"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()

        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["stype"] = swcfg["stype"]

        hwcfg_add = {}
        hwcfg_add["depth"] = hwcfg["depth"]
        hwcfg_add["mode"] = hwcfg["mode"].lower()
        hwcfg_add["scale"] = 2
        hwcfg_add["dima"] = 0
        hwcfg_add["entry"] = None
        self.sadd = FSUAdd(hwcfg_add, swcfg)

    def forward(self, x) -> str:
        return self.sadd(torch.stack([x, torch.ones_like(x)], dim=0))


class HUBHardsigmoid(torch.nn.Module):
    """
    This is a hub scaled addition (x+1)/2.
    """
    def __init__(self, scale=3):
        super(HUBHardsigmoid, self).__init__()
        self.scale = scale

    def forward(self, x) -> str:
        return torch.nn.Hardsigmoid()(x * self.scale)

