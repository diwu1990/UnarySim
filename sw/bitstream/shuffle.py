import torch

class Decorr():
    """
    decorrelate two input bit streams
    """
    def __init__(self, depth=8):
        super(Decorr, self).__init__()
        raise ValueError("Decorr class is not implemented.")

    def Gen(self):
        return None
    

class Desync():
    """
    desynchronize two input bit streams
    """
    def __init__(self, prob, bitwidth=8, mode="unipolar"):
        super(Desync, self).__init__()
        raise ValueError("Desync class is not implemented.")

    def Gen(self):
        return None
    

class Sync():
    """
    synchronize two input bit streams
    """
    def __init__(self, prob, bitwidth=8, mode="unipolar"):
        super(Sync, self).__init__()
        raise ValueError("Sync class is not implemented.")

    def Gen(self):
        return None
    
    
class SkewedSync():
    """
    synchronize two input bit streams
    """
    def __init__(self, depth=2):
        super(SkewedSync, self).__init__()
        self.upper = pow(2, depth) - 1
        self.cnt = torch.zeros(1).type(torch.int8)
        self.out_1 = torch.zeros(1).type(torch.uint8)

    def Gen(self, in_1, in_2):
        # input and output are uint8 type tensor
        # numerator can have larger or same shape as denominator
        # assume input 1 is smaller than input 2, and input 2 is kept unchanged at output
        sum_in = in_1 + in_2
        if list(self.cnt.size()) != list(sum_in.size()):
            self.cnt = torch.zeros(sum_in.size()).type(torch.int8)
        cnt_not_min = torch.ne(self.cnt, 0).type(torch.uint8)
        cnt_not_max = torch.ne(self.cnt, self.upper).type(torch.uint8)

        self.out_1 = in_1.add(torch.eq(sum_in, 1).type(torch.uint8).mul_(cnt_not_min * (1 - in_1) + (0 - cnt_not_max) * in_1))

        self.cnt.add_(torch.eq(sum_in, 1).type(torch.int8).mul_(2 * in_1.type(torch.int8) - 1)).clamp_(0, self.upper)
        return self.out_1, in_2
