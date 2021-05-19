import torch

class SkewedSyncInt(torch.nn.Module):
    """
    synchronize two input bit streams in using integer stochastic computing
    "VLSI Implementation of Deep Neural Network Using Integral Stochastic Computing"
    """
    def __init__(self, 
                 depth=4, 
                 btype=torch.float, 
                 stype=torch.float):
        super(SkewedSyncInt, self).__init__()
        self.btype=btype
        self.stype=stype
        self.upper = torch.nn.Parameter(torch.Tensor([pow(2, depth) - 1]).type(self.btype), requires_grad=False)
        self.cnt = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)
        
    def forward(self, in_1, in_2):
        # input 2 is kept unchanged at output
        # if input 1 is smaller than input 2, this module works the same as SkewedSync
        # if input 1 is larger than input 2, this module aggregates 1s in input 1 to integers larger than 1, and sync them with input 2
        # output 1 is a binary digit stream.
        
        in_2_eq_1 = torch.eq(in_2, 1).type(self.btype)
        # when input 2 is 1, output 1 depends on the sum of the current cnt and the current input 1, otherwise output 1 is 0
        temp_sum = self.cnt + in_1.type(self.btype)
        temp_sum_clip = temp_sum.clamp(0, self.upper.item())
        out_1 = in_2_eq_1 * temp_sum_clip
        self.cnt.data = (temp_sum - out_1).clamp(0, self.upper.item())

        return out_1.type(self.stype), in_2
    