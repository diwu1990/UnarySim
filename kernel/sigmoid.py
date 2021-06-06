import torch
from UnarySim.kernel.add import FSUAdd


class FSUHardsigmoid(torch.nn.Module):
    """
    This is a scaled addition (x+1)/2, compatible to for both unipolar and bipolar bitstream.
    """
    def __init__(self, mode="bipolar", depth=8):
        super(FSUHardsigmoid, self).__init__()
        self.sadd = FSUAdd(mode=mode, scaled=True, dim=0, depth=depth)

    def forward(self, x) -> str:
        return self.sadd(torch.stack([x, torch.ones_like(x)], dim=0))


class ScaleHardsigmoid(torch.nn.Module):
    """
    This is a scaled addition (x+1)/2.
    """
    def __init__(self, scale=3):
        super(ScaleHardsigmoid, self).__init__()
        self.scale = scale

    def forward(self, x) -> str:
        return torch.nn.Hardsigmoid()(x * self.scale)
