import torch

class Decorr():
    """
    decorrelate two input bit streams
    """
    def __init__(self, depth=8):
        super(Decorr, self).__init__()
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
    

class Desync():
    """
    desynchronize two input bit streams
    """
    def __init__(self, prob, bitwidth=8, mode="unipolar"):
        super(Desync, self).__init__()
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
    

class Sync():
    """
    synchronize two input bit streams
    """
    def __init__(self, prob, bitwidth=8, mode="unipolar"):
        super(Sync, self).__init__()
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
    
    
class SkewedSync():
    """
    synchronize two input bit streams
    """
    def __init__(self, prob, bitwidth=8, mode="unipolar"):
        super(SkewedSync, self).__init__()
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