import torch
from UnarySim.sw.stream.gen import RNG, SourceGen, BSGen

class tanh_combinational(torch.nn.Module):
    """
    this module is for combinational tanh. The module is able to compute tanh(ax), where a = 1 in this implementation.
    the detail can be found at "K. Parhi and Y. Liu. 2017. Computing Arithmetic Functions Using Stochastic Logic by Series Expansion. Transactions on Emerging Topics in Computing (2017).", fig.10.
    """
    def __init__(self, 
                 depth=8, 
                 mode="unipolar", 
                 rng="Sobol", 
                 rng_dim=1, 
                 rtype=torch.float,
                 stype=torch.float,
                 btype=torch.float):
        super(tanh_combinational, self).__init__()

        self.bitwidth = depth
        self.mode = mode
        self.rng = rng
        self.rng_dim = rng_dim
        self.rtype = rtype
        self.stype = stype
        self.btype = btype

        # If a unit is named as n_x, it will be used in the calculation of n_x+1.
        self.rng_2 = RNG(
            bitwidth=self.bitwidth,
            dim=self.rng_dim,
            rng=self.rng,
            rtype=self.rtype)()

        self.rng_3 = RNG(
            bitwidth=self.bitwidth,
            dim=self.rng_dim+1,
            rng=self.rng,
            rtype=self.rtype)()

        self.rng_4 = RNG(
            bitwidth=self.bitwidth,
            dim=self.rng_dim+2,
            rng=self.rng,
            rtype=self.rtype)()

        self.rng_5 = RNG(
            bitwidth=self.bitwidth,
            dim=self.rng_dim+3,
            rng=self.rng,
            rtype=self.rtype)()    
        
        # constants used in computation
        self.n2_c = torch.tensor([62/153]).type(self.rtype)
        self.sg_n2_c = SourceGen(self.n2_c, self.bitwidth, self.mode, self.rtype)()
        self.bs_n2_c = BSGen(self.sg_n2_c, self.rng_2, self.stype)
        # self.id_n1_c = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

        self.n3_c = torch.tensor([17/42]).type(self.rtype)
        self.sg_n3_c = SourceGen(self.n3_c, self.bitwidth, self.mode, self.rtype)()
        self.bs_n3_c = BSGen(self.sg_n3_c, self.rng_3, self.stype)
        # self.id_n2_c = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

        self.n4_c = torch.tensor([2/5]).type(self.rtype)
        self.sg_n4_c = SourceGen(self.n4_c, self.bitwidth, self.mode, self.rtype)()
        self.bs_n4_c = BSGen(self.sg_n4_c, self.rng_4, self.stype)
        # self.id_n3_c = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

        self.n5_c = torch.tensor([1/3]).type(self.rtype)
        self.sg_n5_c = SourceGen(self.n5_c, self.bitwidth, self.mode, self.rtype)()
        self.bs_n5_c = BSGen(self.sg_n5_c, self.rng_5, self.stype)
        # self.id_n4_c = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

        self.bs_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

        # print(self.bs_idx)
        # dff to prevent correlation
        
        # 4 dff in series
        self.input_4d1_1 = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)
        self.input_4d2_1 = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)
        self.input_4d3_1 = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)
        self.input_4d4_1 = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)

        self.d1 = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)
        self.d2= torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)
        self.d3 = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)
        
        # 4 dff in series
        self.input_4d1_2 = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)
        self.input_4d2_2 = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)
        self.input_4d3_2 = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)
        self.input_4d4_2 = torch.nn.Parameter(torch.zeros(1).type(self.btype), requires_grad=False)


    def tanh_combinational_forward(self, input):
        
        input_x = input.type(self.stype)

        n1 = (input_x & self.input_4d4_1)

        # Operating units
        n_2_c = self.bs_n2_c(self.bs_idx).type(self.stype)
        n_2 = 1 - (n_2_c & n1)
        
        n_3_c = self.bs_n3_c(self.bs_idx).type(self.stype)
        n_3 = 1 - (n_2 & n_3_c & self.d1)
        
        n_4_c = self.bs_n4_c(self.bs_idx).type(self.stype)
        n_4 = 1 - (n_3 & n_4_c & self.d2)
        
        n_5_c = self.bs_n5_c(self.bs_idx).type(self.stype)
        n_5 = 1 - (n_4 & n_5_c & self.d3)
        
        # print("buffer 3:", self.input_d4)
        output = (n_5 & self.input_4d4_2).type(self.stype)
        

        # Update buffers and idx
        self.d3.data = self.d2
        self.d2.data = self.d1
        self.d1.data = n1
        
        self.input_4d4_2.data = self.input_4d3_2
        self.input_4d3_2.data = self.input_4d2_2
        self.input_4d2_2.data = self.input_4d1_2
        self.input_4d1_2.data = self.input_4d4_1
        self.input_4d4_1.data = self.input_4d3_1
        self.input_4d3_1.data = self.input_4d2_1
        self.input_4d2_1.data = self.input_4d1_1
        self.input_4d1_1.data = input_x
        
      
        # self.id_n1_c.data = self.id_n1_c.add(1)
        # self.id_n2_c.data = self.id_n2_c.add(1)
        # self.id_n3_c.data = self.id_n3_c.add(1)
        # self.id_n4_c.data = self.id_n4_c.add(1)
        self.bs_idx.data = self.bs_idx.add(1)
        # print(self.input_d2)
        # print(self.input_d3)
        # print(self.input_d4)
        # print(self.bs_idx)

        return output
        
    def forward(self, input_x):

        return self.tanh_combinational_forward(input_x).type(self.stype)