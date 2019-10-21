import torch

class RNG(torch.nn.Module):
    """
    Random number generator to generate one random sequence
    """
    def __init__(self, bitwidth=8, dim=1, mode="Sobol"):
        super(RNG, self).__init__()
        self.dim = dim
        self.mode = mode
        self.seq_len = pow(2, bitwidth)
        self.rng_seq = torch.nn.Parameter(torch.Tensor(1, self.seq_len), requires_grad=False)
        if self.mode == "Sobol":
            # get the requested dimension of sobol random number
            self.rng_seq.data = torch.quasirandom.SobolEngine(self.dim).draw(self.seq_len)[:, dim-1].view(self.seq_len).mul_(self.seq_len).type(torch.long)
        elif self.mode == "Race":
            self.rng_seq.data = torch.tensor([x/self.seq_len for x in range(self.seq_len)]).mul_(self.seq_len).type(torch.long)
        else:
            raise ValueError("RNG mode is not implemented.")

    def forward(self):
        return self.rng_seq
    

class RawScale(torch.nn.Module):
    """
    Scale raw data to source data to unary computing, which meets bipolar/unipolar requirements.
    """
    def __init__(self, raw, mode="bipolar", percentile=100):
        super(RawScale, self).__init__()
        self.raw = raw
        self.mode = mode
        
        # to do: add the percentile based scaling
        self.percentile = percentile
        
        self.source = torch.nn.Parameter(torch.Tensor(raw.size()), requires_grad=False)

    def forward(self):
        if self.mode == "unipolar":
            self.source.data = (self.raw - torch.min(self.raw))/(torch.max(self.raw) - torch.min(self.raw))
        elif self.mode == "bipolar":
            self.source.data = (self.raw - torch.min(self.raw))/(torch.max(self.raw) - torch.min(self.raw)) * 2 - 1
        else:
            raise ValueError("SourceGen mode is not implemented.")
        return self.source
    
    
class SourceGen(torch.nn.Module):
    """
    Convert source problistic data to binary integer data
    """
    def __init__(self, prob, bitwidth=8, mode="bipolar"):
        super(SourceGen, self).__init__()
        self.prob = prob
        self.mode = mode
        self.len = pow(2, bitwidth)
        self.binary = torch.nn.Parameter(torch.Tensor(prob.size()), requires_grad=False)
        if mode == "unipolar":
            self.binary.data = self.prob.mul(self.len).round().type(torch.long)
        elif mode == "bipolar":
            self.binary.data = self.prob.add(1).div(2).mul(self.len).round().type(torch.long)
        else:
            raise ValueError("SourceGen mode is not implemented.")

    def forward(self):
        return self.binary
    

class BSGen(torch.nn.Module):
    """
    Compare source data with rng_seq[rng_idx] to generate bit stream from source
    """
    def __init__(self, source, rng_seq):
        super(BSGen, self).__init__()
        self.source = source
        self.rng_seq = rng_seq
    
    def forward(self, rng_idx):
        return torch.gt(self.source, self.rng_seq[rng_idx]).type(torch.int8)
    
    
# class BSRegen(torch.nn.Module):
#     """
#     collect input bit, compare buffered binary with rng_seq[rng_idx] to regenerate bit stream
#     """
#     def __init__(self, depth, rng_seq, mode="unipolar"):
#         super(BSRegen, self).__init__()
#         # self.in_shape = in_shape
#         self.rng_seq = rng_seq
#         self.half = pow(2,depth-1)
#         self.upper = pow(2,depth)-1
#         # self.cnt = torch.ones(in_shape).mul_(self.half).type(torch.long)
#         self.cnt = self.half
    
#     def forward(self, in_bit, rng_idx):
#         self.cnt = self.cnt + in_bit.type(torch.float).mul_(2).sub_(1)
#         self.cnt.type(torch.long).clamp_(0, self.upper)
#         # self.cnt.add_(in_bit.type(torch.long).mul_(2).sub_(1)).clamp_(0, self.upper)
#         return torch.gt(self.cnt, self.rng_seq[rng_idx]).type(torch.int8)
    