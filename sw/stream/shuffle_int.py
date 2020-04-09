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
        self.cnt = torch.nn.Parameter(torch.zeros(1).type(self.buftype), requires_grad=False)
        
    def forward(self, in_1, in_2):
        # input 2 is kept unchanged at output
        # if input 1 is smaller than input 2, this module works the same as SkewedSync
        # if input 1 is larger than input 2, this module aggregates 1s in input 1 to integers larger than 1, and sync them with input 2
        # output 1 is a binary digit stream.
        
        in_2_eq_1 = torch.eq(in_2, 1).type(self.buftype)
        # when input 2 is 1, output 1 depends on the sum of the current cnt and the current input 1, otherwise output 1 is 0
        temp_sum = self.cnt + in_1.type(self.buftype)
        temp_sum_gt_max = torch.gt(temp_sum, self.upper.item()).type(self.buftype)
        out_1 = in_2_eq_1 * (temp_sum_gt_max * self.upper + (1 - temp_sum_gt_max) * temp_sum)
        self.cnt.data = (temp_sum - out_1).clamp(0, self.upper.item())

        return out_1.type(self.stype), in_2
    