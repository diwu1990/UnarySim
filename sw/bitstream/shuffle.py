import torch

class Decorr(torch.nn.Module):
    """
    decorrelate two input bit streams
    """
    def __init__(self, depth=8):
        super(Decorr, self).__init__()
        raise ValueError("Decorr class is not implemented.")

    def forward(self):
        return None
    

class Desync(torch.nn.Module):
    """
    desynchronize two input bit streams
    """
    def __init__(self, prob, bitwidth=8, mode="bipolar"):
        super(Desync, self).__init__()
        raise ValueError("Desync class is not implemented.")

    def forward(self):
        return None
    

class Sync(torch.nn.Module):
    """
    synchronize two input bit streams
    """
    def __init__(self, prob, bitwidth=8, mode="bipolar"):
        super(Sync, self).__init__()
        raise ValueError("Sync class is not implemented.")

    def forward(self):
        return None
    
    
class SkewedSync(torch.nn.Module):
    """
    synchronize two input bit streams in a skewed way
    "in-stream stochastic division and square root via correlation"
    all tensors are torch.nn.Parameter type so as to move to GPU for computing
    """
    def __init__(self, depth=2, bstype=torch.float, buftype=torch.float):
        super(SkewedSync, self).__init__()
        self.buftype=buftype
        self.bstype=bstype
        self.upper = torch.nn.Parameter(torch.Tensor([pow(2, depth) - 1]).type(self.buftype), requires_grad=False)
        self.cnt = torch.nn.Parameter(torch.Tensor(1).type(self.buftype), requires_grad=False)
        self.out_1 = torch.nn.Parameter(torch.Tensor(1).type(self.bstype), requires_grad=False)
        self.cnt_not_min = torch.nn.Parameter(torch.Tensor(1).type(self.bstype), requires_grad=False)
        self.cnt_not_max = torch.nn.Parameter(torch.Tensor(1).type(self.bstype), requires_grad=False)
        
    def forward(self, in_1, in_2):
        # input and output are int8 type tensor
        # numerator can have larger or same shape as denominator
        # assume input 1 is smaller than input 2, and input 2 is kept unchanged at output
        sum_in = in_1 + in_2
        if list(self.cnt.size()) != list(sum_in.size()):
            self.cnt.data = torch.zeros_like(sum_in).type(self.buftype)
        self.cnt_not_min.data = torch.ne(self.cnt, 0).type(self.bstype)
        self.cnt_not_max.data = torch.ne(self.cnt, self.upper.item()).type(self.bstype)

        self.out_1.data = in_1.add(torch.eq(sum_in, 1).type(self.bstype).mul_(self.cnt_not_min * (1 - in_1) + (0 - self.cnt_not_max) * in_1))

        self.cnt.data.add_(torch.eq(sum_in, 1).type(self.buftype).mul_(in_1.mul(2).sub(1))).clamp_(0, self.upper.item())
        return self.out_1, in_2
