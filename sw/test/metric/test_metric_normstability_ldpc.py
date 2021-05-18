# %%
%load_ext autoreload
%autoreload 2
%matplotlib inline

# %%
import torch
import math
from UnarySim.sw.stream.gen import RNG, SourceGen, BSGen
from UnarySim.sw.kernel.add import GainesAdd
from UnarySim.sw.kernel.shiftreg import ShiftReg
from UnarySim.sw.metric.metric import ProgressiveError, NormStability
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import time
import math
import numpy as np

# %%
class CheckNode(torch.nn.Module):
    def __init__(self,
                 degree=4,
                 stype=torch.float):
        super(CheckNode, self).__init__()
        self.stype = stype
        self.c2v = torch.nn.Parameter(torch.ones(degree, 1).type(stype), requires_grad=False)
        self.parity_check = torch.nn.Parameter(torch.ones(1).type(stype), requires_grad=False)

    def forward(self, v2c):
        # assume check node input is stack along axis 0
        self.parity_check.data = (torch.sum(v2c, 0) % 2).type(self.stype)
        self.c2v.data = (torch.sum(v2c, 0).sub(v2c) % 2).type(self.stype)
        return self.c2v, self.parity_check


class VariableNodeCNT(torch.nn.Module):
    def __init__(self,
                 degree=1,
                 depth=7,
                 LLR=None,
                 rtype=torch.float,
                 btype=torch.float, 
                 stype=torch.float):
        super(VariableNodeCNT, self).__init__()
        
        # this degree includes channel information
        self.degree = degree
        assert degree >= 1, "Input degree can't be smaller than 2."
        if degree == 1 or degree == 2:
            # no shift register is required, as the channel information is directly sent to the check node
            pass
        elif degree == 3:
            self.im_0 = ShiftReg(depth=2, stype=stype)
            self.im_1 = ShiftReg(depth=2, stype=stype)
            self.im_2 = ShiftReg(depth=2, stype=stype)
        elif degree == 4:
            self.im_0_0 = ShiftReg(depth=2, stype=stype)
            self.im_0_1 = ShiftReg(depth=2, stype=stype)
            self.im_1_0 = ShiftReg(depth=2, stype=stype)
            self.im_1_1 = ShiftReg(depth=2, stype=stype)
            self.im_2_0 = ShiftReg(depth=2, stype=stype)
            self.im_2_1 = ShiftReg(depth=2, stype=stype)
            self.im_3_0 = ShiftReg(depth=2, stype=stype)
            self.im_3_1 = ShiftReg(depth=2, stype=stype)

        self.acc = torch.nn.Parameter(torch.zeros(1).type(btype), requires_grad=False)
        self.acc.data = LLR.type(btype)
        self.acc_max = 2**depth - 1
        self.acc_max_1 = 2**depth
        
        self.rtype = rtype
        self.btype = btype
        self.stype = stype
    
    def degree1_forward(self, c2v, chn):
        # c2v/v2c is [0][...]
        # chn/posterior is [...]
        c2v_eq = torch.zeros_like(c2v)
        c2v_eq[0] = chn.type(self.btype)
        v2c = c2v_eq.type(self.stype)
        posterior = torch.eq(v2c[0], c2v[0]).type(self.stype)
        return v2c, posterior
    
    def degree2_forward(self, c2v, chn):
        # c2v/v2c is [0, 1][...]
        # chn/posterior is [...]
        # index 0
        c2v_eq_0 = (1 - (chn.type(torch.int8) ^ c2v[1].type(torch.int8))).type(self.btype)
        # index 1
        c2v_eq_1 = (1 - (chn.type(torch.int8) ^ c2v[0].type(torch.int8))).type(self.btype)
        c2v_eq = torch.stack((c2v_eq_0, c2v_eq_1), 0)
        v2c = c2v_eq.type(self.stype) \
            * chn \
            + (1 - c2v_eq).type(self.stype) \
            * (torch.gt(self.acc, torch.randint(0, self.acc_max_1, (self.degree, 1)).type(self.btype))).type(self.stype)
        self.acc.data = (self.acc + c2v_eq * chn.mul(2).sub(1).type(self.btype)).clamp(0, self.acc_max)
        posterior = torch.eq(v2c[0], c2v[0]).type(self.stype)
        return v2c, posterior
    
    def degree3_forward(self, c2v, chn):
        # c2v/v2c is [0, 1, 2][...]
        # chn/posterior is [...]
        # index 0
        c2v_eq_0_0 = (1 - (chn.type(torch.int8) ^ c2v[1].type(torch.int8))).type(self.btype)
        internal_0 = c2v_eq_0_0.type(self.stype) \
                   * chn \
                   + (1 - c2v_eq_0_0).type(self.stype) \
                   * self.im_0.sr.data[torch.randint(0, 2, (1, )).type(torch.long).item()]
        dc0, dc1 = self.im_0(internal_0, mask = c2v_eq_0_0)
        c2v_eq_0 = (1 - (internal_0.type(torch.int8) ^ c2v[2].type(torch.int8))).type(self.btype)
        
        # index 1
        c2v_eq_1_0 = (1 - (chn.type(torch.int8) ^ c2v[0].type(torch.int8))).type(self.btype)
        internal_1 = c2v_eq_1_0.type(self.stype) \
                   * chn \
                   + (1 - c2v_eq_1_0).type(self.stype) \
                   * self.im_1.sr.data[torch.randint(0, 2, (1, )).type(torch.long).item()]
        dc0, dc1 = self.im_1(internal_1, mask = c2v_eq_1_0)
        c2v_eq_1 = (1 - (internal_1.type(torch.int8) ^ c2v[2].type(torch.int8))).type(self.btype)
        
        # index 2
        c2v_eq_2_0 = (1 - (chn.type(torch.int8) ^ c2v[0].type(torch.int8))).type(self.btype)
        internal_2 = c2v_eq_2_0.type(self.stype) \
                   * chn \
                   + (1 - c2v_eq_2_0).type(self.stype) \
                   * self.im_2.sr.data[torch.randint(0, 2, (1, )).type(torch.long).item()]
        dc0, dc1 = self.im_2(internal_0, mask = c2v_eq_2_0)
        c2v_eq_2 = (1 - (internal_2.type(torch.int8) ^ c2v[1].type(torch.int8))).type(self.btype)
        
        c2v_eq = torch.stack((c2v_eq_0, c2v_eq_1, c2v_eq_2), 0)
        input_1 =  torch.stack((internal_0, internal_1, internal_2), 0)
        v2c = c2v_eq.type(self.stype) \
            * input_1 \
            + (1 - c2v_eq).type(self.stype) \
            * (torch.gt(self.acc, torch.randint(0, self.acc_max_1, (self.degree, 1)).type(self.btype))).type(self.stype)
        self.acc.data = (self.acc + c2v_eq * input_1.mul(2).sub(1).type(self.btype)).clamp(0, self.acc_max)
        posterior = torch.eq(v2c[0], c2v[0]).type(self.stype)
        return v2c, posterior
    
    def degree4_forward(self, c2v, chn):
        # c2v/v2c is [0, 1, 2, 3][...]
        # chn/posterior is [...]
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # index 0
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # chn, c2v[1]
        c2v_eq_0_0 = (1 - (chn.type(torch.int8) ^ c2v[1].type(torch.int8))).type(self.btype)
        internal_0_0 = c2v_eq_0_0.type(self.stype) \
                     * chn \
                     + (1 - c2v_eq_0_0).type(self.stype) \
                     * self.im_0_0.sr.data[torch.randint(0, 2, (1, )).type(torch.long).item()]
        dc0, dc1 = self.im_0_0(internal_0_0, mask = c2v_eq_0_0)
        # c2v[2], c2v[3]
        c2v_eq_0_1 = (1 - (c2v[2].type(torch.int8) ^ c2v[3].type(torch.int8))).type(self.btype)
        internal_0_1 = c2v_eq_0_1.type(self.stype) \
                     * c2v[2] \
                     + (1 - c2v_eq_0_1).type(self.stype) \
                     * self.im_0_1.sr.data[torch.randint(0, 2, (1, )).type(torch.long).item()]
        dc0, dc1 = self.im_0_1(internal_0_1, mask = c2v_eq_0_1)
        
        c2v_eq_0 = (1 - (internal_0_0.type(torch.int8) ^ internal_0_1.type(torch.int8))).type(self.btype)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # index 1
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # chn, c2v[0]
        c2v_eq_1_0 = (1 - (chn.type(torch.int8) ^ c2v[0].type(torch.int8))).type(self.btype)
        internal_1_0 = c2v_eq_1_0.type(self.stype) \
                     * chn \
                     + (1 - c2v_eq_1_0).type(self.stype) \
                     * self.im_1_0.sr.data[torch.randint(0, 2, (1, )).type(torch.long).item()]
        dc0, dc1 = self.im_1_0(internal_1_0, mask = c2v_eq_1_0)
        # c2v[2], c2v[3]
        c2v_eq_1_1 = (1 - (c2v[2].type(torch.int8) ^ c2v[3].type(torch.int8))).type(self.btype)
        internal_1_1 = c2v_eq_1_1.type(self.stype) \
                     * c2v[2] \
                     + (1 - c2v_eq_1_1).type(self.stype) \
                     * self.im_1_1.sr.data[torch.randint(0, 2, (1, )).type(torch.long).item()]
        dc0, dc1 = self.im_1_1(internal_1_1, mask = c2v_eq_1_1)
        
        c2v_eq_1 = (1 - (internal_1_0.type(torch.int8) ^ internal_1_1.type(torch.int8))).type(self.btype)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # index 2
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # chn, c2v[0]
        c2v_eq_2_0 = (1 - (chn.type(torch.int8) ^ c2v[0].type(torch.int8))).type(self.btype)
        internal_2_0 = c2v_eq_2_0.type(self.stype) \
                     * chn \
                     + (1 - c2v_eq_2_0).type(self.stype) \
                     * self.im_2_0.sr.data[torch.randint(0, 2, (1, )).type(torch.long).item()]
        dc0, dc1 = self.im_2_0(internal_2_0, mask = c2v_eq_2_0)
        # c2v[1], c2v[3]
        c2v_eq_2_1 = (1 - (c2v[1].type(torch.int8) ^ c2v[3].type(torch.int8))).type(self.btype)
        internal_2_1 = c2v_eq_2_1.type(self.stype) \
                     * c2v[1] \
                     + (1 - c2v_eq_2_1).type(self.stype) \
                     * self.im_2_1.sr.data[torch.randint(0, 2, (1, )).type(torch.long).item()]
        dc0, dc1 = self.im_2_1(internal_2_1, mask = c2v_eq_2_1)
        
        c2v_eq_2 = (1 - (internal_2_0.type(torch.int8) ^ internal_2_1.type(torch.int8))).type(self.btype)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # index 3
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # chn, c2v[0]
        c2v_eq_3_0 = (1 - (chn.type(torch.int8) ^ c2v[0].type(torch.int8))).type(self.btype)
        internal_3_0 = c2v_eq_3_0.type(self.stype) \
                     * chn \
                     + (1 - c2v_eq_3_0).type(self.stype) \
                     * self.im_3_0.sr.data[torch.randint(0, 2, (1, )).type(torch.long).item()]
        dc0, dc1 = self.im_3_0(internal_3_0, mask = c2v_eq_3_0)
        # c2v[1], c2v[2]
        c2v_eq_3_1 = (1 - (c2v[1].type(torch.int8) ^ c2v[2].type(torch.int8))).type(self.btype)
        internal_3_1 = c2v_eq_3_1.type(self.stype) \
                     * c2v[1] \
                     + (1 - c2v_eq_3_1).type(self.stype) \
                     * self.im_3_1.sr.data[torch.randint(0, 2, (1, )).type(torch.long).item()]
        dc0, dc1 = self.im_3_1(internal_3_1, mask = c2v_eq_3_1)
        
        c2v_eq_3 = (1 - (internal_3_0.type(torch.int8) ^ internal_3_1.type(torch.int8))).type(self.btype)
        
        c2v_eq = torch.stack((c2v_eq_0, c2v_eq_1, c2v_eq_2, c2v_eq_3), 0)
        input_1 =  torch.stack((internal_0_0, internal_1_0, internal_2_0, internal_3_0), 0)
        v2c = c2v_eq.type(self.stype) \
            * input_1 \
            + (1 - c2v_eq).type(self.stype) \
            * (torch.gt(self.acc, torch.randint(0, self.acc_max_1, (self.degree, 1)).type(self.btype))).type(self.stype)
        self.acc.data = (self.acc + c2v_eq * input_1.mul(2).sub(1).type(self.btype)).clamp(0, self.acc_max)
        posterior = torch.eq(v2c[0], c2v[0]).type(self.stype)
        return v2c, posterior
    
    def forward(self, c2v, chn):
        if self.degree == 1:
            v2c, posterior = self.degree1_forward(c2v, chn)
        elif self.degree == 2:
            v2c, posterior = self.degree2_forward(c2v, chn)
        elif self.degree == 3:
            v2c, posterior = self.degree3_forward(c2v, chn)
        elif self.degree ==4:
            v2c, posterior = self.degree4_forward(c2v, chn)
        return v2c.type(self.stype), posterior.type(self.stype)


# %%
cn0 = CheckNode()
cn0_out, parity_check = cn0(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
print(cn0_out)
print(parity_check)

# %%
vn0 = VariableNodeCNT(degree=1, depth=7, LLR=torch.tensor([0.7*(2**7)]))
vn0_out, posterior = vn0(torch.tensor([1.]), torch.tensor([1.]))
print(vn0_out, posterior)
vn0_out, posterior = vn0(torch.tensor([0.]), torch.tensor([0.]))
print(vn0_out, posterior)

# %%
vn0 = VariableNodeCNT(degree=2, depth=7, LLR=torch.tensor([0.7*(2**7)]))
vn0_out, posterior = vn0(torch.tensor([[1.], [1.]]), torch.tensor([1.]))
print(vn0_out, posterior)
vn0_out, posterior = vn0(torch.tensor([[0.], [0.]]), torch.tensor([0.]))
print(vn0_out, posterior)

# %%
vn0 = VariableNodeCNT(degree=3, depth=7, LLR=torch.tensor([0.7*(2**7)]))
vn0_out, posterior = vn0(torch.tensor([[1.], [1.], [1.]]), torch.tensor([1.]))
print(vn0_out, posterior)
vn0_out, posterior = vn0(torch.tensor([[0.], [0.], [0.]]), torch.tensor([0.]))
print(vn0_out, posterior)

# %%
vn0 = VariableNodeCNT(degree=4, depth=7, LLR=torch.tensor([0.7*(2**7)]))
vn0_out, posterior = vn0(torch.tensor([[1.], [1.], [1.], [1.]]), torch.tensor([1.]))
print(vn0_out, posterior)
vn0_out, posterior = vn0(torch.tensor([[0.], [0.], [0.], [0.]]), torch.tensor([0.]))
print(vn0_out, posterior)

# %%
def test(depth=7, 
         bitwidth=7, 
         rng="Sobol",
         rand_idx=1, 
         threshold=0.05
        ):
    G = torch.tensor(
        [
            [1., 0., 0., 1., 0., 1.],
            [0., 1., 0., 1., 1., 1.],
            [0., 0., 1., 1., 1., 0.]
        ]
    )
    
    H = torch.tensor(
        [
            [1., 1., 1., 1., 0., 0.],
            [0., 0., 1., 1., 0., 1.],
            [1., 0., 0., 1., 1., 0.]
        ]
    )
    print("G*H_t: ", G.matmul(H.t())%2)
    
    src = torch.tensor([1., 0., 1.])
    
    encoded = src.matmul(G) % 2
    print("encoded: ", encoded)
    
    parity_check_result = encoded.matmul(H.t())%2
    print("parity_check_result", parity_check_result)
    
    channel = encoded + torch.randn(6)*2 - 1
    channel = encoded
    print("channel: ", channel)
    
    a = 0.9
    Y = 6
    LLR = 4 * a / Y * channel
    
    e = 2.7182818284
    factor = 1
    chn_prob = torch.pow(e, LLR) / (torch.pow(e, LLR) + 1) * factor
    print("chn_prob: ", chn_prob)
    
    bitwidth = 8
    mode = "unipolar"
    depth = 7

    chn_probRNG = RNG(bitwidth, rand_idx, rng)()
    # print("chn_probRNG: ", chn_probRNG)
    chn_probSRC = SourceGen(chn_prob, bitwidth, mode=mode)()
    chn_probBSGen = BSGen(chn_probSRC, chn_probRNG)

    CN0 = CheckNode(degree=4)
    CN1 = CheckNode(degree=3)
    CN2 = CheckNode(degree=3)
    
    VN0 = VariableNodeCNT(degree=2, depth=depth, LLR=torch.tensor([chn_prob[0] * (2**depth)]))
    VN1 = VariableNodeCNT(degree=1, depth=depth, LLR=torch.tensor([chn_prob[1] * (2**depth)]))
    VN2 = VariableNodeCNT(degree=2, depth=depth, LLR=torch.tensor([chn_prob[2] * (2**depth)]))
    VN3 = VariableNodeCNT(degree=3, depth=depth, LLR=torch.tensor([chn_prob[3] * (2**depth)]))
    VN4 = VariableNodeCNT(degree=1, depth=depth, LLR=torch.tensor([chn_prob[4] * (2**depth)]))
    VN5 = VariableNodeCNT(degree=1, depth=depth, LLR=torch.tensor([chn_prob[5] * (2**depth)]))
    
    chn_probNS = NormStability(chn_prob, mode=mode, threshold=threshold)
    posteriorNS = NormStability(src, mode=mode, threshold=threshold)
    posteriorCNT = torch.zeros_like(src)
    
    with torch.no_grad():
        for i in range(2**bitwidth):
#         for i in range(2):
            chn_prob_bs = chn_probBSGen(torch.tensor([i]))
            chn_probNS.Monitor(chn_prob_bs)
            
            VN0_c2v = torch.stack((CN0.c2v[0].view(1, ), 
                                   CN2.c2v[0].view(1, )), 0)
            VN1_c2v = CN0.c2v[1].view(1, ).unsqueeze(0)
            VN2_c2v = torch.stack((CN0.c2v[2].view(1, ), 
                                   CN1.c2v[0].view(1, )), 0)
            VN3_c2v = torch.stack((CN0.c2v[3].view(1, ), 
                                   CN1.c2v[1].view(1, ), 
                                   CN2.c2v[1].view(1, )), 0)
            VN4_c2v = CN2.c2v[2].view(1, ).unsqueeze(0)
            VN5_c2v = CN1.c2v[2].view(1, ).unsqueeze(0)
#             print("VN0_c2v: \n", VN0_c2v)
#             print("VN1_c2v: \n", VN1_c2v)
#             print("VN2_c2v: \n", VN2_c2v)
#             print("VN3_c2v: \n", VN3_c2v)
#             print("VN4_c2v: \n", VN4_c2v)
#             print("VN5_c2v: \n", VN5_c2v)
            
            VN0_chn = chn_prob_bs[0].view((1, ))
            VN1_chn = chn_prob_bs[1].view((1, ))
            VN2_chn = chn_prob_bs[2].view((1, ))
            VN3_chn = chn_prob_bs[3].view((1, ))
            VN4_chn = chn_prob_bs[4].view((1, ))
            VN5_chn = chn_prob_bs[5].view((1, ))
            
            VN0_v2c, VN0_posterior = VN0(VN0_c2v, VN0_chn)
            VN1_v2c, VN1_posterior = VN1(VN1_c2v, VN1_chn)
            VN2_v2c, VN2_posterior = VN2(VN2_c2v, VN2_chn)
            VN3_v2c, VN3_posterior = VN3(VN3_c2v, VN3_chn)
            VN4_v2c, VN4_posterior = VN4(VN4_c2v, VN4_chn)
            VN5_v2c, VN5_posterior = VN5(VN5_c2v, VN5_chn)
            
            CN0_v2c = torch.stack((VN0_v2c[0].view(1, ), 
                                   VN1_v2c[0].view(1, ), 
                                   VN2_v2c[0].view(1, ), 
                                   VN3_v2c[0].view(1, )), 0)
            CN1_v2c = torch.stack((VN2_v2c[1].view(1, ), 
                                   VN3_v2c[1].view(1, ), 
                                   VN5_v2c[0].view(1, )), 0)
            CN2_v2c = torch.stack((VN0_v2c[1].view(1, ), 
                                   VN3_v2c[2].view(1, ), 
                                   VN4_v2c[0].view(1, )), 0)
            
            CN0_c2v, CN0_parity_check = CN0(CN0_v2c)
            CN1_c2v, CN1_parity_check = CN1(CN1_v2c)
            CN2_c2v, CN2_parity_check = CN2(CN2_v2c)
            
            parity_check_sum = CN0_parity_check + CN1_parity_check + CN2_parity_check
            
            if parity_check_sum == 0:
                print(i, "-th cycle decode success!!!!!!!!!!!!")
            else:
                print(i, "-th cycle")
            posterior_bs = torch.stack((VN0_posterior, VN1_posterior, VN2_posterior), 0).squeeze()
            posteriorCNT = posteriorCNT + posterior_bs * 2 -1
            posteriorNS.Monitor(posterior_bs)
            print(torch.lt(posteriorCNT, 0).type(torch.float) == src)
            
    
    

# %%
output = test()

# %%
# all success from 112 the cycle,
# thus total norm stab is
norm_stab_all = 1 - 111/256
print("norm_stab_all:", norm_stab_all)

# vn0 succeed at 112 cycle
norm_stab_vn0 = 1 - 111/256
print("norm_stab_vn0:", norm_stab_vn0)

# vn1 first succeed at 86 cycle
norm_stab_vn1 = 1 - 85/256
print("norm_stab_vn1:", norm_stab_vn1)

# vn2 first succeed at 64 cycle
norm_stab_vn2 = 1 - 63/256
print("norm_stab_vn2:", norm_stab_vn2)

# %%
