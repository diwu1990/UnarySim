import torch
import numpy as np

class RNG(torch.nn.Module):
    """
    Random number generator to generate one random sequence, returns a tensor of size [2**bitwidth]
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
    

class RNGMulti(torch.nn.Module):
    """
    Random number generator to generate multiple random sequences, returns a tensor of size [dim, 2**bitwidth]
    """
    def __init__(self, bitwidth=8, dim=1, mode="Sobol", transpose=False):
        super(RNGMulti, self).__init__()
        self.dim = dim
        self.mode = mode
        self.seq_len = pow(2, bitwidth)
        self.rng_seq = torch.nn.Parameter(torch.Tensor(1, self.seq_len), requires_grad=False)
        if self.mode == "Sobol":
            # get the requested dimension of sobol random number
            self.rng_seq.data = torch.quasirandom.SobolEngine(self.dim).draw(self.seq_len).mul_(self.seq_len).type(torch.long)
        else:
            raise ValueError("RNG mode is not implemented.")
        if transpose is True:
            self.rng_seq.transpose(0, 1)

    def forward(self):
        return self.rng_seq
    

class RawScale(torch.nn.Module):
    """
    Scale raw data to source data in unary computing, which meets bipolar/unipolar requirements.
    input percentile should be a number in range (0, 100].
    """
    def __init__(self, raw, mode="bipolar", percentile=100):
        super(RawScale, self).__init__()
        self.raw = raw
        self.mode = mode
        
        # to do: add the percentile based scaling
        self.percentile_down = (100 - percentile)/2
        self.percentile_up = 100 - self.percentile_down
        self.clamp_min = np.percentile(raw.cpu(), self.percentile_down)
        self.clamp_max = np.percentile(raw.cpu(), self.percentile_up)

        self.source = torch.nn.Parameter(torch.Tensor(raw.size()), requires_grad=False)
        self.source.data = raw.clamp(self.clamp_min, self.clamp_max)

    def forward(self):
        if self.mode == "unipolar":
            self.source.data = (self.source - torch.min(self.source))/(torch.max(self.source) - torch.min(self.source))
        elif self.mode == "bipolar":
            self.source.data = (self.source - torch.min(self.source))/(torch.max(self.source) - torch.min(self.source)) * 2 - 1
        else:
            raise ValueError("RawScale mode is not implemented.")
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
    
    
class BSGenMulti(torch.nn.Module):
    """
    Compare source data with rng_seq indexed with rng_idx to generate bit streams from source
    """
    def __init__(self, source, rng_seq, dim=0):
        super(BSGenMulti, self).__init__()
        self.source = source
        self.rng_seq = rng_seq
        self.dim = dim
    
    def forward(self, rng_idx):
        return torch.gt(self.source, torch.gather(self.rng_seq, self.dim, rng_idx)).type(torch.int8)
    
