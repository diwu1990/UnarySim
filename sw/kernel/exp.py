import torch
from UnarySim.sw.stream.gen import RNG, SourceGen, BSGen

class expN1(torch.nn.Module):
    """
    This module is for combinational exp(-x)
    the detail can be found at "K. Parhi and Y. Liu. 2017. Computing Arithmetic Functions Using Stochastic Logic by Series Expansion. Transactions on Emerging Topics in Computing (2017).", fig.12.
    """
    def __init__(self, 
                 mode="unipolar", 
                 rng="Sobol", 
                 rng_dim=1, 
                 rng_width=8, 
                 rtype=torch.float, 
                 stype=torch.float, 
                 btype=torch.float):
        super(expN1, self).__init__()

        self.bitwidth = rng_width
        self.mode = mode
        self.rng = rng
        self.rng_dim = rng_dim
        self.rtype = rtype
        self.stype = stype
        self.btype = btype
        
        assert mode == "unipolar", "Combinational expN1 needs unipolar mode."
        self.rng_1 = RNG(bitwidth=self.bitwidth, dim=self.rng_dim+0, rng=self.rng, rtype=self.rtype)()
        self.rng_2 = RNG(bitwidth=self.bitwidth, dim=self.rng_dim+1, rng=self.rng, rtype=self.rtype)()
        self.rng_3 = RNG(bitwidth=self.bitwidth, dim=self.rng_dim+2, rng=self.rng, rtype=self.rtype)()
        self.rng_4 = RNG(bitwidth=self.bitwidth, dim=self.rng_dim+3, rng=self.rng, rtype=self.rtype)()

        # constants used in computation
        self.n1_c = torch.tensor([0.2000]).type(self.rtype)
        self.n2_c = torch.tensor([0.2500]).type(self.rtype)
        self.n3_c = torch.tensor([0.3333]).type(self.rtype)
        self.n4_c = torch.tensor([0.5000]).type(self.rtype)

        self.sg_n1_c = SourceGen(self.n1_c, self.bitwidth, self.mode, self.rtype)()
        self.sg_n2_c = SourceGen(self.n2_c, self.bitwidth, self.mode, self.rtype)()
        self.sg_n3_c = SourceGen(self.n3_c, self.bitwidth, self.mode, self.rtype)()
        self.sg_n4_c = SourceGen(self.n4_c, self.bitwidth, self.mode, self.rtype)()

        self.bs_n1_c = BSGen(self.sg_n1_c, self.rng_1, self.stype)
        self.bs_n2_c = BSGen(self.sg_n2_c, self.rng_2, self.stype)
        self.bs_n3_c = BSGen(self.sg_n3_c, self.rng_3, self.stype)
        self.bs_n4_c = BSGen(self.sg_n4_c, self.rng_4, self.stype)

        # dff as isolation to mitigate correlation
        self.input_d1 = torch.nn.Parameter(torch.zeros(1).type(self.stype), requires_grad=False)
        self.input_d2 = torch.nn.Parameter(torch.zeros(1).type(self.stype), requires_grad=False)
        self.input_d3 = torch.nn.Parameter(torch.zeros(1).type(self.stype), requires_grad=False)
        self.input_d4 = torch.nn.Parameter(torch.zeros(1).type(self.stype), requires_grad=False)

        self.bs_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

    def exp_comb_forward(self, input):
        # Operating units
        n_1_c = self.bs_n1_c(self.bs_idx)
        n_2_c = self.bs_n2_c(self.bs_idx)
        n_3_c = self.bs_n3_c(self.bs_idx)
        n_4_c = self.bs_n4_c(self.bs_idx)

        n_1 = 1 - (input.type(torch.int8) & n_1_c.type(torch.int8))
        n_2 = 1 - (n_1 & n_2_c.type(torch.int8) & self.input_d1.type(torch.int8))
        n_3 = 1 - (n_2 & n_3_c.type(torch.int8) & self.input_d2.type(torch.int8))
        n_4 = 1 - (n_3 & n_4_c.type(torch.int8) & self.input_d3.type(torch.int8))
        
        output = 1 - (n_4 & self.input_d4.type(torch.int8))

        # Update buffers and idx
        self.input_d4.data = self.input_d3
        self.input_d3.data = self.input_d2
        self.input_d2.data = self.input_d1
        self.input_d1.data = input

        self.bs_idx.data = self.bs_idx + 1
        return output
        
    def forward(self, input):
        return self.exp_comb_forward(input).type(self.stype)


class expNG(torch.nn.Module):
    """
    This module is for fsm exp(-2Gx).
    Input is bipolar, output is unipolar.
    "Stochastic neural computation I: Computational elements"
    """
    def __init__(self, 
                 mode="bipolar", 
                 depth=5, 
                 gain=1, 
                 rtype=torch.float, 
                 stype=torch.float, 
                 btype=torch.float):
        super(expNG, self).__init__()

        self.depth = depth
        self.mode = mode
        self.rtype = rtype
        self.stype = stype
        self.btype = btype
        
        assert mode == "bipolar", "FSM expNG needs bipolar mode."
        # N is the number of state, G is 1
        self.max = torch.nn.Parameter(torch.tensor([2**depth-1]).type(self.btype), requires_grad=False)
        self.thd = torch.nn.Parameter(torch.tensor([2**depth-gain]).type(self.btype), requires_grad=False)
        self.cnt = torch.nn.Parameter(torch.tensor([2**(depth-1)]).type(self.btype), requires_grad=False)

    def exp_fsm_forward(self, input):
        output = torch.zeros_like(input)
        output = output + torch.lt(self.cnt, self.thd.item()).type(self.stype)
        self.cnt.data = input.type(self.btype) * (self.cnt + 1) + (1 - input.type(self.btype)) * (self.cnt - 1)
        self.cnt.data = self.cnt.clamp(0, self.max.item())
        return output
        
    def forward(self, input):
        return self.exp_fsm_forward(input)
    
    