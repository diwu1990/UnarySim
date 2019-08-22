import torch

class RNG(object):
    """
    random number generator class
    generate one random sequence
    """
    def __init__(self, bitwidth=8, dim=1, mode="Sobol"):
        super(RNG, self).__init__()
        self.bitwidth = bitwidth
        self.dim = dim
        self.mode = mode
        self.seq_len = pow(2,self.bitwidth)

        if self.mode == "Sobol":
            temp_seq = torch.quasirandom.SobolEngine(self.dim).draw(self.seq_len)
            # get the requested dimension of sobol random number
            self.rng_seq = temp_seq[:, dim-1].view(self.seq_len).mul_(self.seq_len).type(torch.long)
        elif self.mode == "Race":
            self.rng_seq = torch.tensor([x/self.seq_len for x in range(self.seq_len)]).mul_(self.seq_len).type(torch.long)
        else:
            raise ValueError("RNG mode is not implemented.")
        
    def Out(self):
        return self.rng_seq
    