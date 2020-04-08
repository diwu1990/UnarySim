import torch
import math
from UnarySim.sw.stream.gen import RNG, SourceGen, BSGen

class Correlation(torch.nn.Module):
    """
    calculate the correlation between two input bit streams.
    SC correlation: "Exploiting correlation in stochastic circuit design"
    """
    def __init__(self):
        super(Correlation, self).__init__()
        self.paired_00_d = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.paired_01_c = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.paired_10_b = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.paired_11_a = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.len = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.in_1_d = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def Monitor(self, in_1, in_2=None):
        if in_2 is None:
            in_2 = self.in_1_d.clone().detach()
            self.in_1_d.data = in_1.clone().detach()
        in_1_is_0 = torch.eq(in_1, 0).type(torch.float)
        in_1_is_1 = 1 - in_1_is_0
        in_2_is_0 = torch.eq(in_2, 0).type(torch.float)
        in_2_is_1 = 1 - in_2_is_0
        self.paired_00_d.data.add_(in_1_is_0 * in_2_is_0)
        self.paired_01_c.data.add_(in_1_is_0 * in_2_is_1)
        self.paired_10_b.data.add_(in_1_is_1 * in_2_is_0)
        self.paired_11_a.data.add_(in_1_is_1 * in_2_is_1)
        self.len.data.add_(1)
    
    def forward(self):
        ad_minus_bc = self.paired_11_a * self.paired_00_d - self.paired_10_b * self.paired_01_c
        ad_gt_bc = torch.gt(ad_minus_bc, 0).type(torch.float)
        ad_le_bc = 1 - ad_gt_bc
        a_plus_b = self.paired_11_a + self.paired_10_b
        a_plus_c = self.paired_11_a + self.paired_01_c
        a_minus_d = self.paired_11_a - self.paired_00_d
        all_0_tensor = torch.zeros_like(self.paired_11_a)
        all_1_tensor = torch.ones_like(self.paired_11_a)
        corr_ad_gt_bc = ad_minus_bc.div(
            torch.max(
                torch.min(a_plus_b, a_plus_c).mul_(self.len).sub_(a_plus_b.mul(a_plus_c)), 
                all_1_tensor
            )
        )
        corr_ad_le_bc = ad_minus_bc.div(
            torch.max(
                a_plus_b.mul(a_plus_c).sub(torch.max(a_minus_d, all_0_tensor).mul_(self.len)),
                all_1_tensor
            )
        )
        return ad_gt_bc * corr_ad_gt_bc + ad_le_bc * corr_ad_le_bc
    
    
class ProgressiveError(torch.nn.Module):
    """
    calculate progressive error based on progressive precision of input bit stream.
    progressive precision: "Fast and accurate computation using stochastic circuits"
    scale=1 indicates non-scale, scale>1 indicates scale.
    """
    def __init__(self, in_value, scale=1, mode="bipolar"):
        super(ProgressiveError, self).__init__()
        # in_value is always binary
        # after scaling, unipolar should be within (0, 1), bipolar should be within (-1, 1).
        # therefore, clamping with (-1, 1) always works
        self.in_value = torch.clamp(in_value/scale, -1., 1.)
        self.mode = mode
        assert self.mode is "unipolar" or self.mode is "bipolar", "ProgressiveError mode is not implemented."
        self.len = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.one_cnt = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.out_pp = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.err = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def Monitor(self, in_1):
        self.one_cnt.data = self.one_cnt.data.add(in_1.type(torch.float))
        self.len.data.add_(1)

    def forward(self):
        self.out_pp.data = self.one_cnt.div(self.len)
        if self.mode is "bipolar":
            self.out_pp.data = self.out_pp.mul(2).sub(1)
        self.err.data = self.out_pp.sub(self.in_value)
        return self.out_pp, self.err
    
    
class Stability(torch.nn.Module):
    """
    calculate the stability of one bit stream.
    stability: "uGEMM: Unary Computing Architecture for GEMM Applications"
    """
    def __init__(self, in_value, mode="bipolar", threshold=0.05):
        super(Stability, self).__init__()
        self.in_value = in_value
        self.mode = mode
        self.threshold = torch.nn.Parameter(torch.tensor([threshold]), requires_grad=False)
        self.len = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.err = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.stable_len = torch.zeros_like(in_value)
        self.stability = torch.zeros_like(in_value)
        self.pp = ProgressiveError(in_value, scale=1, mode=mode)
        
    def Monitor(self, in_1):
        self.pp.Monitor(in_1)
        self.len = self.pp.len
        _, self.err = self.pp()
        self.stable_len.add_(torch.gt(self.err.abs(), self.threshold).type(torch.float).mul_(self.len - self.stable_len))
        
    def forward(self):
        self.stability = 1 - self.stable_len.clamp(1, self.len.item()).div(self.len)
        return self.stability
    

def search_best_stab(P_low_L, P_high_L, L):
    """
    This function is used to search the best stability length, R and l_p
    """
    max_stab_len = L
    max_stab_R = 1
    max_stab_l_p = 1
    for p_L in range(P_low_L, P_high_L+1):
        l_p = L/math.gcd(p_L, L)

        # one more bit 0
        B_L = 0
        if p_L == P_low_L:
            R_low = 1 if P_low_L <= B_L else L
        else:
            R_low = (P_low_L - B_L)/l_p/(p_L - P_low_L)

        if P_high_L == p_L:
            R_high = 1 if B_L <= P_high_L else L
        else:
            R_high = (B_L - P_high_L)/l_p/(P_high_L - p_L)

        R_0 = math.ceil(max(R_low, R_high))

        # one more bit 1
        B_L = L
        if p_L == P_low_L:
            R_low = 1 if P_low_L <= B_L else L
        else:
            R_low = (P_low_L - B_L)/l_p/(p_L - P_low_L)

        if P_high_L == p_L:
            R_high = 1 if B_L <= P_high_L else L
        else:
            R_high = (B_L - P_high_L)/l_p/(P_high_L - p_L)

        R_L = math.ceil(max(R_low, R_high))

        R = min(R_0, R_L)

        if R*l_p < max_stab_len:
            max_stab_len = max(R*l_p, 1)
            max_stab_R = R
            max_stab_l_p = l_p
    return max_stab_len, max_stab_R, max_stab_l_p


class NormStability(torch.nn.Module):
    """
    calculate the normalized value-independent stability, which is standard stability over maximum stability.
    normalized stability is acutual stability/max possible stability
    """
    def __init__(self, in_value, mode="bipolar", threshold=0.05):
        super(NormStability, self).__init__()
        self.in_value = in_value
        self.mode = mode
        self.threshold = torch.nn.Parameter(torch.tensor([threshold]), requires_grad=False)
        self.stability = Stability(in_value, mode=mode, threshold=threshold)
        self.min_prob = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.max_prob = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode is "bipolar":
            self.min_prob.data = torch.max((in_value + 1) / 2 - threshold / 2, torch.zeros_like(in_value))
            self.max_prob.data = torch.min((in_value + 1) / 2 + threshold / 2, torch.ones_like(in_value))
        else:
            self.min_prob.data = torch.max(in_value - threshold, torch.zeros_like(in_value))
            self.max_prob.data = torch.min(in_value + threshold, torch.ones_like(in_value))
        self.len = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.in_shape = in_value.size()
        self.max_stab = torch.zeros_like(in_value)
        self.max_stab_len = torch.ones_like(in_value)
        self.max_stab_l_p = torch.ones_like(in_value)
        self.max_stab_R = torch.ones_like(in_value)

    def Monitor(self, in_1):
        self.stability.Monitor(in_1)
        self.len = self.stability.len
        
    def forward(self):
        assert self.len != 0, "Input bit stream length can't be 0."
        dim = len(self.in_shape)
        assert dim <= 4, "Input dimension larger than 4 is not implemented."
        L = pow(2, math.ceil(math.log2(self.len)))
        # use ceil for lower to avoid 0
        P_low_L_all = torch.floor(self.min_prob*L).clamp(0, L)
        # use ceil for upper to avoid the case that upper is smaller than lower, when bit stream length is small
        P_high_L_all = torch.ceil(self.max_prob*L).clamp(0, L)
        
        if dim == 1:
            for index_0 in range(self.in_shape[0]):
                P_low_L = P_low_L_all[index_0].type(torch.long).item()
                P_high_L = P_high_L_all[index_0].type(torch.long).item()
                max_stab_len, max_stab_R, max_stab_l_p = search_best_stab(P_low_L, P_high_L, L)
                self.max_stab[index_0] = max(1 - max_stab_len/self.len, 0)
                self.max_stab_len[index_0] = max_stab_len
                self.max_stab_l_p[index_0] = max_stab_l_p
                self.max_stab_R[index_0] = max_stab_R
                
        if dim == 2:
            for index_0 in range(self.in_shape[0]):
                for index_1 in range(self.in_shape[1]):
                    P_low_L = P_low_L_all[index_0][index_1].type(torch.long).item()
                    P_high_L = P_high_L_all[index_0][index_1].type(torch.long).item()
                    max_stab_len, max_stab_R, max_stab_l_p = search_best_stab(P_low_L, P_high_L, L)
                    self.max_stab[index_0][index_1] = max(1 - max_stab_len/self.len, 0)
                    self.max_stab_len[index_0][index_1] = max_stab_len
                    self.max_stab_l_p[index_0][index_1] = max_stab_l_p
                    self.max_stab_R[index_0][index_1] = max_stab_R
                
        if dim == 3:
            for index_0 in range(self.in_shape[0]):
                for index_1 in range(self.in_shape[1]):
                    for index_2 in range(self.in_shape[2]):
                        P_low_L = P_low_L_all[index_0][index_1][index_2].type(torch.long).item()
                        P_high_L = P_high_L_all[index_0][index_1][index_2].type(torch.long).item()
                        max_stab_len, max_stab_R, max_stab_l_p = search_best_stab(P_low_L, P_high_L, L)
                        self.max_stab[index_0][index_1][index_2] = max(1 - max_stab_len/self.len, 0)
                        self.max_stab_len[index_0][index_1][index_2] = max_stab_len
                        self.max_stab_l_p[index_0][index_1][index_2] = max_stab_l_p
                        self.max_stab_R[index_0][index_1][index_2] = max_stab_R
                        
        if dim == 4:
            for index_0 in range(self.in_shape[0]):
                for index_1 in range(self.in_shape[1]):
                    for index_2 in range(self.in_shape[2]):
                        for index_3 in range(self.in_shape[3]):
                            P_low_L = P_low_L_all[index_0][index_1][index_2][index_3].type(torch.long).item()
                            P_high_L = P_high_L_all[index_0][index_1][index_2][index_3].type(torch.long).item()
                            max_stab_len, max_stab_R, max_stab_l_p = search_best_stab(P_low_L, P_high_L, L)
                            self.max_stab[index_0][index_1][index_2][index_3] = max(1 - max_stab_len/self.len, 0)
                            self.max_stab_len[index_0][index_1][index_2][index_3] = max_stab_len
                            self.max_stab_l_p[index_0][index_1][index_2][index_3] = max_stab_l_p
                            self.max_stab_R[index_0][index_1][index_2][index_3] = max_stab_R
        
        normstab = self.stability()/self.max_stab
        normstab[torch.isnan(normstab)] = 0
        # some normstab is larger than 1.0,
        # as the method based on the segmented uniform,
        # which is an approximation of the best case
        normstab.clamp_(0, 1)
        
        return normstab
    
def gen_ns_out(new_ns_len, out_cnt_ns, out_cnt_st, ns_gen, st_gen, bs_st, bs_ns, L):
    """
    This function is used to generate the output for NSbuilder
    """
    if out_cnt_ns == new_ns_len:      
        ns_gen = False
        st_gen = True
        
    if out_cnt_st == L:
        st_gen = False

    if ns_gen == True:
        output = bs_ns(out_cnt_ns)
        out_cnt_ns = out_cnt_ns + 1    
        if out_cnt_ns == L:
            print("In gry stage, reach max bs length")
        
    if st_gen == True:
        output = bs_st(out_cnt_st)
        out_cnt_st = out_cnt_st + 1

    return out_cnt_ns, out_cnt_st, ns_gen, st_gen, output


class NSbuilder(torch.nn.Module):
    """
    this module is the normalized stability builder. 
    it converts the normalized stability of a bitstream into the desired value.
    """
    def __init__(self,
                 depth=8,
                 mode="bipolar",
                 normstability=0.5,
                 threshold=0.05,
                 value=None,
                 rng_dim=1,
                 rng="Sobol",
                 bstype=torch.float,
                 randtype=torch.float):
        super(NSbuilder, self).__init__()
        
        self.depth = depth
        self.length = torch.tensor([2**self.depth]).type(torch.float)
        self.normstb = normstability
        self.val = value
        self.mode = mode
        self.val_shape = self.val.size()
        self.val_dim = len(self.val_shape)

        self.bstype = bstype
        self.randtype = randtype

        self.T = threshold
        self.L = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
        self.lp = torch.zeros_like(self.val)
        self.R = torch.ones_like(self.val)

        self.P_low = torch.zeros_like(self.val)
        self.P_up = torch.zeros_like(self.val)

        self.max_stable = torch.zeros_like(self.val)
        self.max_st_len = torch.zeros_like(self.val)
        self.new_st_len = torch.zeros_like(self.val)
        self.new_ns_len = torch.zeros_like(self.val)
        self.new_ns_upp = torch.zeros_like(self.val)
        self.new_ns_low = torch.zeros_like(self.val)
        
        self.new_ns_val = torch.zeros_like(self.val)
        self.new_st_val = torch.zeros_like(self.val)
        self.new_ns_one = torch.zeros_like(self.val)
        self.new_st_one = torch.zeros_like(self.val)

        self.rng = RNG(
            bitwidth=depth,
            dim=rng_dim,
            rng=rng,
            randtype=randtype)()

        self.src_st = None
        self.src_ns = None
        self.bs_st = None
        self.bs_ns = None

        self.ns_gen = torch.ones_like(self.val).type(torch.bool)
        self.st_gen = torch.zeros_like(self.val).type(torch.bool)
        
        self.out_cnt_ns = torch.zeros_like(self.val).type(torch.float)
        self.out_cnt_st = torch.zeros_like(self.val).type(torch.float)

        self.output = torch.zeros_like(self.val).type(torch.float)

        ## INIT:
        # Stage to calculate several essential params
        self.L.data = 2**(torch.ceil(torch.log2(self.length)))
        self.P_low.data = torch.max(self.val - self.T, torch.zeros_like(self.val))
        self.P_up.data = torch.min(torch.ones_like(self.val), self.val + self.T)
        upper = torch.min(torch.ceil(self.L * self.P_up), self.L.type(torch.float))
        lower = torch.max(torch.floor(self.L * self.P_low), torch.zeros_like(self.L))

        if self.val_dim == 1:
            for index_0 in range(self.val_shape[0]):
                P_low_L = lower[index_0].type(torch.long).item()
                P_high_L = upper[index_0].type(torch.long).item()
                L = self.L.type(torch.long).item()
                max_stab_len, max_stab_R, max_stab_l_p = search_best_stab(P_low_L, P_high_L, L)
                self.max_stable[index_0] = max_stab_len
                self.lp[index_0] = max_stab_l_p
                self.R[index_0] = max_stab_R

            self.max_st_len = self.L - (self.max_stable)
            self.new_st_len.data = torch.ceil(self.max_st_len * self.normstb)  
            self.new_ns_len.data = (self.L - self.new_st_len)

            for index_0 in range(self.val_shape[0]):  
                if self.val[index_0].item() > 0.5:
                    self.new_ns_one[index_0] = (self.P_up[index_0])*(self.new_ns_len[index_0]+1)
                else:
                    self.new_ns_one[index_0] = torch.max((self.P_low[index_0]*(self.new_ns_len[index_0]+1)-1),torch.zeros_like(self.L))
        
        if self.val_dim == 2:
            for index_0 in range(self.val_shape[0]):
                for index_1 in range(self.val_shape[1]):
                    P_low_L = lower[index_0][index_1].type(torch.long).item()
                    P_high_L = upper[index_0][index_1].type(torch.long).item()
                    L = self.L.type(torch.long).item()
                    max_stab_len, max_stab_R, max_stab_l_p = search_best_stab(P_low_L, P_high_L, L)
                    self.max_stable[index_0][index_1] = max_stab_len
                    self.lp[index_0][index_1] = max_stab_l_p
                    self.R[index_0][index_1] = max_stab_R
                    
            self.max_st_len = self.L - (self.max_stable)
            self.new_st_len.data = torch.ceil(self.max_st_len * self.normstb)  
            self.new_ns_len.data = (self.L - self.new_st_len)

            for index_0 in range(self.val_shape[0]):
                for index_1 in range(self.val_shape[1]):
                    if self.val[index_0][index_1].item() > 0.5:
                        self.new_ns_one[index_0][index_1] = (self.P_up[index_0][index_1])*(self.new_ns_len[index_0][index_1]+1)
                    else:
                        self.new_ns_one[index_0][index_1] = torch.max((self.P_low[index_0][index_1]*(self.new_ns_len[index_0][index_1]+1)-1),torch.zeros_like(self.L))
        
        if self.val_dim == 3:
            for index_0 in range(self.val_shape[0]):
                for index_1 in range(self.val_shape[1]):
                    for index_2 in range(self.val_shape[2]):
                        P_low_L = lower[index_0][index_1][index_2].type(torch.long).item()
                        P_high_L = upper[index_0][index_1][index_2].type(torch.long).item()
                        L = self.L.type(torch.long).item()
                        max_stab_len, max_stab_R, max_stab_l_p = search_best_stab(P_low_L, P_high_L, L)
                        self.max_stable[index_0][index_1][index_2] = max_stab_len
                        self.lp[index_0][index_1][index_2] = max_stab_l_p
                        self.R[index_0][index_1][index_2] = max_stab_R

            self.max_st_len = self.L - (self.max_stable)
            self.new_st_len.data = torch.ceil(self.max_st_len * self.normstb)  
            self.new_ns_len.data = (self.L - self.new_st_len)

            for index_0 in range(self.val_shape[0]):
                for index_1 in range(self.val_shape[1]):
                    for index_2 in range(self.val_shape[2]):
                        if self.val[index_0][index_1][index_2].item() > 0.5:
                            self.new_ns_one[index_0][index_1][index_2] = (self.P_up[index_0][index_1][index_2])*(self.new_ns_len[index_0][index_1][index_2]+1)
                        else:
                            self.new_ns_one[index_0][index_1][index_2] = torch.max((self.P_low[index_0][index_1][index_2]*(self.new_ns_len[index_0][index_1][index_2]+1)-1),torch.zeros_like(self.L))
        
        if self.val_dim == 4:
            for index_0 in range(self.val_shape[0]):
                for index_1 in range(self.val_shape[1]):
                    for index_2 in range(self.val_shape[2]):
                        for index_3 in range(self.val_shape[3]):
                            P_low_L = lower[index_0][index_1][index_2][index_3].type(torch.long).item()
                            P_high_L = upper[index_0][index_1][index_2][index_3].type(torch.long).item()
                            L = self.L.type(torch.long).item()
                            max_stab_len, max_stab_R, max_stab_l_p = search_best_stab(P_low_L, P_high_L, L)
                            self.max_stable[index_0][index_1][index_2][index_3] = max_stab_len
                            self.lp[index_0][index_1][index_2][index_3] = max_stab_l_p
                            self.R[index_0][index_1][index_2][index_3] = max_stab_R

            self.max_st_len = self.L - (self.max_stable)
            self.new_st_len.data = torch.ceil(self.max_st_len * self.normstb)  
            self.new_ns_len.data = (self.L - self.new_st_len)

            for index_0 in range(self.val_shape[0]):
                for index_1 in range(self.val_shape[1]):
                    for index_2 in range(self.val_shape[2]):
                        for index_3 in range(self.val_shape[3]):
                            if self.val[index_0][index_1][index_2][index_3].item() > 0.5:
                                self.new_ns_one[index_0][index_1][index_2][index_3] = (self.P_up[index_0][index_1][index_2][index_3])*(self.new_ns_len[index_0][index_1][index_2][index_3]+1)
                            else:
                                self.new_ns_one[index_0][index_1][index_2][index_3] = torch.max((self.P_low[index_0][index_1][index_2][index_3]*(self.new_ns_len[index_0][index_1][index_2][index_3]+1)-1),torch.zeros_like(self.L))

        self.new_st_one.data = (self.val * self.L - self.new_ns_one)
        self.new_ns_val.data = self.new_ns_one / self.new_ns_len   
        self.new_st_val.data = self.new_st_one / self.new_st_len            

        self.src_st = SourceGen(self.new_st_val, self.depth, self.mode, self.randtype)()
        self.src_ns = SourceGen(self.new_ns_val, self.depth, self.mode, self.randtype)()
        self.bs_st = BSGen(self.src_st, self.rng)
        self.bs_ns = BSGen(self.src_ns, self.rng)

    def NSbuilder_forward(self):

        ## Stage to generate output
        if self.val_dim == 1:
            for index_0 in range(self.val_shape[0]):
                new_ns_len = self.new_ns_len[index_0].item()
                out_cnt_ns = self.out_cnt_ns[index_0]
                out_cnt_st = self.out_cnt_st[index_0]
                ns_gen = self.ns_gen[index_0].item()
                st_gen = self.st_gen[index_0].item()
                bs_st = self.bs_st
                bs_ns = self.bs_ns
                L_in = self.L.item()
                out_cnt_ns, out_cnt_st, ns_gen, st_gen, output = gen_ns_out(new_ns_len, out_cnt_ns, out_cnt_st, ns_gen, st_gen, bs_st, bs_ns, L_in)
                self.out_cnt_ns[index_0] = out_cnt_ns
                self.out_cnt_st[index_0] = out_cnt_st
                self.ns_gen[index_0] = ns_gen
                self.st_gen[index_0] = st_gen
                self.output[index_0] = output[index_0].item()
        
        if self.val_dim == 2:
            for index_0 in range(self.val_shape[0]):
                for index_1 in range(self.val_shape[1]):
                    new_ns_len = self.new_ns_len[index_0][index_1].item()
                    out_cnt_ns = self.out_cnt_ns[index_0][index_1]
                    out_cnt_st = self.out_cnt_st[index_0][index_1]
                    ns_gen = self.ns_gen[index_0][index_1].item()
                    st_gen = self.st_gen[index_0][index_1].item()
                    bs_st = self.bs_st
                    bs_ns = self.bs_ns
                    L_in = self.L.item()
                    out_cnt_ns, out_cnt_st, ns_gen, st_gen, output = gen_ns_out(new_ns_len, out_cnt_ns, out_cnt_st, ns_gen, st_gen, bs_st, bs_ns, L_in)
                    self.out_cnt_ns[index_0][index_1] = out_cnt_ns
                    self.out_cnt_st[index_0][index_1] = out_cnt_st
                    self.ns_gen[index_0][index_1] = ns_gen
                    self.st_gen[index_0][index_1] = st_gen
                    self.output[index_0][index_1] = output[index_0][index_1].item()
        
        if self.val_dim == 3:
            for index_0 in range(self.val_shape[0]):
                for index_1 in range(self.val_shape[1]):
                    for index_2 in range(self.val_shape[2]):
                        new_ns_len = self.new_ns_len[index_0][index_1][index_2].item()
                        out_cnt_ns = self.out_cnt_ns[index_0][index_1][index_2]
                        out_cnt_st = self.out_cnt_st[index_0][index_1][index_2]
                        ns_gen = self.ns_gen[index_0][index_1][index_2].item()
                        st_gen = self.st_gen[index_0][index_1][index_2].item()
                        bs_st = self.bs_st
                        bs_ns = self.bs_ns
                        L_in = self.L.item()
                        out_cnt_ns, out_cnt_st, ns_gen, st_gen, output = gen_ns_out(new_ns_len, out_cnt_ns, out_cnt_st, ns_gen, st_gen, bs_st, bs_ns, L_in)
                        self.out_cnt_ns[index_0][index_1][index_2] = out_cnt_ns
                        self.out_cnt_st[index_0][index_1][index_2] = out_cnt_st
                        self.ns_gen[index_0][index_1][index_2] = ns_gen
                        self.st_gen[index_0][index_1][index_2] = st_gen
                        self.output[index_0][index_1][index_2] = output[index_0][index_1][index_2].item()

        if self.val_dim == 4:
            for index_0 in range(self.val_shape[0]):
                for index_1 in range(self.val_shape[1]):
                    for index_2 in range(self.val_shape[2]):
                        for index_3 in range(self.val_shape[3]):
                            new_ns_len = self.new_ns_len[index_0][index_1][index_2][index_3].item()
                            out_cnt_ns = self.out_cnt_ns[index_0][index_1][index_2][index_3]
                            out_cnt_st = self.out_cnt_st[index_0][index_1][index_2][index_3]
                            ns_gen = self.ns_gen[index_0][index_1][index_2][index_3].item()
                            st_gen = self.st_gen[index_0][index_1][index_2][index_3].item()
                            bs_st = self.bs_st
                            bs_ns = self.bs_ns
                            L_in = self.L.item()
                            out_cnt_ns, out_cnt_st, ns_gen, st_gen, output = gen_ns_out(new_ns_len, out_cnt_ns, out_cnt_st, ns_gen, st_gen, bs_st, bs_ns, L_in)
                            self.out_cnt_ns[index_0][index_1][index_2][index_3] = out_cnt_ns
                            self.out_cnt_st[index_0][index_1][index_2][index_3] = out_cnt_st
                            self.ns_gen[index_0][index_1][index_2][index_3] = ns_gen
                            self.st_gen[index_0][index_1][index_2][index_3] = st_gen
                            self.output[index_0][index_1][index_2][index_3] = output[index_0][index_1][index_2][index_3].item()

        return self.output.type(self.bstype)

    def forward(self):
        return self.NSbuilder_forward()
    
