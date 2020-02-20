import torch
from UnarySim.sw.bitstream.gen import RNG, SourceGen, BSGen

class UnaryAdd(torch.nn.Module):
    """
    this module is for unary addition, including scaled/non-scaled, unipolar/bipolar.
    """
    def __init__(self, 
                 bitwidth=8, 
                 mode="bipolar", 
                 scaled=True, 
                 acc_dim=0):
        super(UnaryAdd, self).__init__()
        
        # data bit width
        self.bitwidth = bitwidth
        # data representation
        self.mode = mode
        # whether it is scaled addition
        self.scaled = scaled
        # dimension to do reduce sum
        self.acc_dim = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        self.acc_dim.fill_(acc_dim)
        
        # upper bound for accumulation counter in non-scaled mode
        # it is the number of inputs, e.g., the size along the acc_dim dimension
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.scaled is False:
            self.out_accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input):
        self.acc_bound.fill_(input.size()[self.acc_dim.item()])
        if self.mode is "bipolar":
            self.offset.fill_((self.acc_bound.item()-1)/2)
        self.accumulator.data = self.accumulator.add(torch.sum(input.type(torch.float), self.acc_dim.item()))
        
        if self.scaled is True:
            output = torch.ge(self.accumulator, self.acc_bound).type(torch.float)
            self.accumulator.sub_(output * self.acc_bound)
        else:
            self.accumulator.sub_(self.offset)
            output = torch.gt(self.accumulator, self.out_accumulator).type(torch.float)
            self.out_accumulator.data = self.out_accumulator.add(output)

        return output.type(torch.int8)
        

class GainesAdd(torch.nn.Module):
    """
    this module is for Gaines addition.
    1) MUX for scaled addition
    2) OR gate for non-scaled addition
    """
    def __init__(self, 
                 mode="bipolar", 
                 scaled=True, 
                 acc_dim=0):
        super(GainesAdd, self).__init__()
        
        # data representation
        self.mode = mode
        # whether it is scaled addition
        self.scaled = scaled
        # dimension to do reduce sum
        self.acc_dim = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        self.acc_dim.fill_(acc_dim)
        
    def forward(self, input):
        self.acc_bound.fill_(input.size()[self.acc_dim.item()])
        if self.mode is "bipolar":
            self.offset.fill_((self.acc_bound.item()-1)/2)
        self.accumulator.data = self.accumulator.add(torch.sum(input.type(torch.float), self.acc_dim.item()))
        
        if self.scaled is True:
            output = torch.ge(self.accumulator, self.acc_bound).type(torch.float)
            self.accumulator.sub_(output * self.acc_bound)
        else:
            self.accumulator.sub_(self.offset)
            output = torch.gt(self.accumulator, self.out_accumulator).type(torch.float)
            self.out_accumulator.data = self.out_accumulator.add(output)

        return output.type(torch.int8)