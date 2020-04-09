import torch

class SkewedSyncInt(torch.nn.Module):
    """
    synchronize two input bit streams in using integer stochastic computing
    "VLSI Implementation of Deep Neural Network Using Integral Stochastic Computing"
    """
    def __init__(self, depth=4, stype=torch.float, buftype=torch.float):
        super(SkewedSyncInt, self).__init__()
        self.buftype=buftype
        self.stype=stype
        self.upper = torch.nn.Parameter(torch.Tensor([pow(2, depth) - 1]).type(self.buftype), requires_grad=False)
        self.cnt = torch.nn.Parameter(torch.Tensor(1).type(self.buftype), requires_grad=False)
        self.out_1 = torch.nn.Parameter(torch.Tensor(1).type(self.stype), requires_grad=False)
        self.cnt_not_min = torch.nn.Parameter(torch.Tensor(1).type(self.stype), requires_grad=False)
        self.cnt_not_max = torch.nn.Parameter(torch.Tensor(1).type(self.stype), requires_grad=False)
        
    def forward(self, in_1, in_2):
        # input 2 is kept unchanged at output
        # if input 1 is smaller than input 2, this module works the same as SkewedSync
        # if input 1 is larger than input 2, this module aggregates 1s in input 1 to integers larger than 1, and sync them with input 2
        # output 1 is a binary digit stream.
        in_2_eq_1 = torch.eq(in_2, 1).type(torch.int8)
        sum_in = in_1 + in_2
        if list(self.cnt.size()) != list(sum_in.size()):
            self.cnt.data = torch.zeros_like(sum_in).type(self.buftype)
        self.cnt_not_min.data = torch.ne(self.cnt, 0).type(self.stype)
        self.cnt_not_max.data = torch.ne(self.cnt, self.upper.item()).type(self.stype)

        self.out_1.data = in_1.add(torch.eq(sum_in, 1).type(self.stype).mul_(self.cnt_not_min * (1 - in_1) + (0 - self.cnt_not_max) * in_1))
        self.cnt.data.add_(torch.eq(sum_in, 1).type(self.buftype).mul_(in_1.mul(2).sub(1).type(self.buftype))).clamp_(0, self.upper.item())
        return self.out_1, in_2
    