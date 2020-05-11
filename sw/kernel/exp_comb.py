import torch
from UnarySim.sw.stream.gen import RNG, SourceGen, BSGen

class exp_combinational(torch.nn.Module):
    """
    this module is for combinational exp. The module is able to compute exp(-ax), where a = 1.9 in this implementation.
    """
    def __init__(self, 
                 depth=8, 
                 mode="bipolar", 
                 rng="Sobol", 
                 rng_dim=1, 
                 rtype=torch.float,
                 stype=torch.int8):
        super(exp_combinational, self).__init__()

        self.bitwidth = depth
        self.mode = mode
        self.rng = rng
        self.rng_dim = rng_dim
        self.rtype = rtype
        self.stype = stype

        self.rng_1 = RNG(
            bitwidth=self.bitwidth,
            dim=self.rng_dim,
            rng=self.rng,
            rtype=self.rtype)()

        self.rng_2 = RNG(
            bitwidth=self.bitwidth,
            dim=self.rng_dim+1,
            rng=self.rng,
            rtype=self.rtype)()

        self.rng_3 = RNG(
            bitwidth=self.bitwidth,
            dim=self.rng_dim+2,
            rng=self.rng,
            rtype=self.rtype)()

        self.rng_4 = RNG(
            bitwidth=self.bitwidth,
            dim=self.rng_dim+3,
            rng=self.rng,
            rtype=self.rtype)()    
        
        # constants used in computation
        self.n1_c = torch.tensor([0.2000]).type(self.rtype)
        self.sg_n1_c = SourceGen(self.n1_c, self.bitwidth, self.mode, self.rtype)()
        self.bs_n1_c = BSGen(self.sg_n1_c, self.rng_1, self.stype)
        # self.id_n1_c = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

        self.n2_c = torch.tensor([0.2500]).type(self.rtype)
        self.sg_n2_c = SourceGen(self.n2_c, self.bitwidth, self.mode, self.rtype)()
        self.bs_n2_c = BSGen(self.sg_n2_c, self.rng_2, self.stype)
        # self.id_n2_c = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

        self.n3_c = torch.tensor([0.3333]).type(self.rtype)
        self.sg_n3_c = SourceGen(self.n3_c, self.bitwidth, self.mode, self.rtype)()
        self.bs_n3_c = BSGen(self.sg_n3_c, self.rng_3, self.stype)
        # self.id_n3_c = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

        self.n4_c = torch.tensor([0.5000]).type(self.rtype)
        self.sg_n4_c = SourceGen(self.n4_c, self.bitwidth, self.mode, self.rtype)()
        self.bs_n4_c = BSGen(self.sg_n4_c, self.rng_4, self.stype)
        # self.id_n4_c = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

        self.bs_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)


        # dff to prevent correlation
        self.input_d1 = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        self.input_d2= torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        self.input_d3 = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        self.input_d4 = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)


    def exp_combinational_forward(self, input):
        
        input_x = input.type(torch.int8)

        # Operating units
        n_1_c = self.bs_n1_c(self.bs_idx)
        n_1 = 1 - (n_1_c & input_x)
        
        n_2_c = self.bs_n2_c(self.bs_idx)
        n_2 = 1 - (n_1 & n_2_c & self.input_d1)
        
        n_3_c = self.bs_n3_c(self.bs_idx)
        n_3 = 1 - (n_2 & n_3_c & self.input_d2)
        
        n_4_c = self.bs_n4_c(self.bs_idx)
        n_4 = 1 - (n_3 & n_4_c & self.input_d3)
        
        # print("buffer 3:", self.input_d4)
        output = 1 - (n_4 & self.input_d4)
        

        # Update buffers and idx
        self.input_d4.data = self.input_d3
        self.input_d3.data = self.input_d2
        self.input_d2.data = self.input_d1
        self.input_d1.data = input_x
        
        
      
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

        return self.exp_combinational_forward(input_x).type(self.stype)