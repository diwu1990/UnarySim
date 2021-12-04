import torch

class SkewedSync(torch.nn.Module):
    """
    synchronize two input bitstreams in a skewed way, please refer to
    1) "In-Stream Stochastic Division and Square Root via Correlation"
    2) "In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing"
    """
    def __init__(
        self, 
        hwcfg={
            "depth" : 2
        },
        swcfg={
            "btype" : torch.float, 
            "stype" : torch.float
        }):
        super(SkewedSync, self).__init__()
        self.hwcfg = {}
        self.hwcfg["depth"] = hwcfg["depth"]

        self.swcfg = {}
        self.swcfg["stype"] = swcfg["stype"]
        self.swcfg["btype"] = swcfg["btype"]

        self.depth=hwcfg["depth"]
        self.btype=swcfg["btype"]
        self.stype=swcfg["stype"]
        self.upper = torch.nn.Parameter(torch.Tensor([2**self.depth - 1]).type(self.btype), requires_grad=False)
        self.cnt = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)
        
    def forward(self, in_1, in_2):
        # assume input 1 is smaller than input 2, and input 2 is kept unchanged at output
        sum_in = in_1 + in_2
        if list(self.cnt.size()) != list(sum_in.size()):
            self.cnt.data = torch.zeros_like(sum_in).type(self.btype)
        cnt_not_min = torch.ne(self.cnt, 0).type(self.stype)
        cnt_not_max = torch.ne(self.cnt, self.upper.item()).type(self.stype)

        out_1 = in_1.add(torch.eq(sum_in, 1).type(self.stype).mul_(cnt_not_min * (1 - in_1) + (0 - cnt_not_max) * in_1))
        self.cnt.data.add_(torch.eq(sum_in, 1).type(self.btype).mul_(in_1.mul(2).sub(1).type(self.btype))).clamp_(0, self.upper.item())
        return out_1, in_2


class Bi2Uni(torch.nn.Module):
    """
    Convert bipolar bitstreams to unipolar with non-scaled addition, please refer to
    "In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing"
    """
    def __init__(
        self, 
        hwcfg={
            "depth" : 3
        },
        swcfg={
            "stype" : torch.float,
            "btype" : torch.float
        }):
        super(Bi2Uni, self).__init__()
        self.hwcfg = {}
        self.hwcfg["depth"] = hwcfg["depth"]

        self.swcfg = {}
        self.swcfg["stype"] = swcfg["stype"]
        self.swcfg["btype"] = swcfg["btype"]

        self.depth = hwcfg["depth"]
        self.stype = swcfg["stype"]
        self.btype = swcfg["btype"]
        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False).type(self.btype)
        # max value in the accumulator
        self.acc_max = 2**(self.depth-2)
        # min value in the accumulator
        self.acc_min = -2**(self.depth-2)

    def forward(self, input):
        # calculate (2*input-1)/1
        # input bitstreams are [input, input, 0]
        self.accumulator.data = self.accumulator.add(input.type(self.btype)*2 - 1).clamp(self.acc_min, self.acc_max)
        output = torch.ge(self.accumulator, 1).type(self.btype)
        self.accumulator.sub_(output).clamp(self.acc_min, self.acc_max)
        return output.type(self.stype)


class Uni2Bi(torch.nn.Module):
    """
    Convert unipolar bitstreams to bipolar with scaled addition, please refer to
    "In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing"
    """
    def __init__(
        self,
        hwcfg={
            "depth" : 4
        },
        swcfg={
            "stype" : torch.float,
            "btype" : torch.float
        }):
        super(Uni2Bi, self).__init__()
        self.hwcfg = {}
        self.hwcfg["depth"] = hwcfg["depth"]

        self.swcfg = {}
        self.swcfg["stype"] = swcfg["stype"]
        self.swcfg["btype"] = swcfg["btype"]

        self.depth = hwcfg["depth"]
        self.stype = swcfg["stype"]
        self.btype = swcfg["btype"]
        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False).type(self.btype)
        # max value in the accumulator
        self.acc_max = 2**(self.depth-2)
        # min value in the accumulator
        self.acc_min = -2**(self.depth-2)

    def forward(self, input):
        # calculate (input+1)/2
        # input bitstreams are [input, 1]
        input_stack = torch.stack([input.type(self.btype), torch.ones_like(input, dtype=self.btype)], dim=0)
        self.accumulator.data = self.accumulator.add(torch.sum(input_stack.type(self.btype), 0)).clamp(self.acc_min, self.acc_max)
        output = torch.ge(self.accumulator, 2).type(self.btype)
        self.accumulator.sub_(output * 2).clamp(self.acc_min, self.acc_max)
        return output.type(self.stype)

