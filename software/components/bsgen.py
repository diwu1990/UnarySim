import torch

class SourceGen():
    """
    convert source problistic data to binary integer data
    """
    def __init__(self, prob, bitwidth=8, mode="unipolar"):
        super(SourceGen, self).__init__()
        self.prob = prob
        self.bitwidth = bitwidth
        self.mode = mode
        self.len = pow(2,bitwidth)
        if mode == "unipolar":
            self.binary = self.prob.mul_(self.len).round_().type(torch.long)
        elif mode == "bipolar":
            self.binary = self.prob.add_(1).div_(2).mul_(self.len).round_().type(torch.long)
        else:
            raise ValueError("SourceGen mode is not implemented.")

    def Gen(self):
        return self.binary
    

class BSGen(object):
    """
    compare source data with rng_seq[rng_idx] to generate bit stream from source
    """
    def __init__(self, source, rng_seq):
        super(BSGen, self).__init__()
        self.source = source
        self.rng_seq = rng_seq
    
    def Gen(self, rng_idx):
        return torch.gt(self.source, self.rng_seq[rng_idx]).type(torch.uint8)

    
class BSRegen(object):
    """
    collect input bit, compare buffered binary with rng_seq[rng_idx] to regenerate bit stream
    """
    def __init__(self, input_shape, depth, rng_seq):
        super(BSRegen, self).__init__()
        self.input_shape = input_shape
        self.rng_seq = rng_seq
        self.half = pow(2,depth-1)
        self.upper = pow(2,depth)-1
        self.cnt = torch.ones(input_shape).mul_(self.half).type(torch.long)
    
    def Gen(self, in_bit, rng_idx):
        self.cnt.add_(in_bit.type(torch.long).mul_(2).sub_(1)).clamp_(0, self.upper)
        return torch.gt(self.cnt, self.rng_seq[rng_idx]).type(torch.uint8)