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
    def __init__(self, 
                 depth=2, 
                 btype=torch.float, 
                 stype=torch.float):
        super(SkewedSync, self).__init__()
        self.btype=btype
        self.stype=stype
        self.upper = torch.nn.Parameter(torch.Tensor([pow(2, depth) - 1]).type(btype), requires_grad=False)
        self.cnt = torch.nn.Parameter(torch.zeros(1).type(btype), requires_grad=False)
        
    def forward(self, in_1, in_2):
        # assume input 1 is smaller than input 2, and input 2 is kept unchanged at output
        # output 1 is a unary bit stream.
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
    Format conversion from bipolar to unipolar with unary non-scaled addition
    input need to be larger than 0
    """
    def __init__(self, 
                 stype=torch.float):
        super(Bi2Uni, self).__init__()
        self.stype = stype
        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.out_accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input):
        # calculate 2*input-1, this is the resultant unipolar bit stream.
        # input_stack = torch.stack([input, input, torch.zeros_like(input)], dim=0)
        # parallel counter
        # self.accumulator.data = self.accumulator.add(torch.sum(input_stack.type(torch.float), 0))
        
        # following code has the same result as previous
        self.accumulator.data = self.accumulator.add(input*2)
        # offset substraction
        self.accumulator.sub_(1)
        # output generation
        output = torch.gt(self.accumulator, self.out_accumulator).type(torch.float)
        self.out_accumulator.data = self.out_accumulator.add(output)
        return output.type(self.stype)
    
    
class Uni2Bi(torch.nn.Module):
    """
    Format conversion from unipolar to bipolar with unary scaled addition
    """
    def __init__(self, 
                 stype=torch.float):
        super(Uni2Bi, self).__init__()
        self.stype = stype
        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input):
        input_stack = torch.stack([input, torch.ones_like(input)], dim=0)
        self.accumulator.data = self.accumulator.add(torch.sum(input_stack.type(torch.float), 0))
        output = torch.ge(self.accumulator, 2).type(torch.float)
        self.accumulator.sub_(output * 2)
        return output.type(self.stype)
        