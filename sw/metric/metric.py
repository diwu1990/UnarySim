import torch
import math
import numpy as np
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


def search_best_stab_parallel_numpy(P_low_L, P_high_L, L, seach_range):
    """
    This function is used to search the best stability length, R and l_p
    All data are numpy arrays
    """
    max_stab_len = np.ones_like(P_low_L)
    max_stab_len.fill(L.item(0))
    max_stab_R = np.ones_like(P_low_L)
    max_stab_l_p = np.ones_like(P_low_L)
    
    for i in range(seach_range.item(0) + 1):
        p_L = np.clip(P_low_L+i, None, P_high_L)
        l_p = L/np.gcd(p_L, L)
        
        l_p_by_p_L_minus_P_low_L = l_p * i
        l_p_by_P_high_L_min_p_L = l_p * (P_high_L - p_L)
        l_p_by_p_L_minus_P_low_L[l_p_by_p_L_minus_P_low_L==0] = 1
        l_p_by_P_high_L_min_p_L[l_p_by_P_high_L_min_p_L==0] = 1

        # one more bit 0
        B_L = 0
        p_L_eq_P_low_L = (p_L == P_low_L).astype("float32")
        P_low_L_le_B_L = (P_low_L <= B_L).astype("float32")
        R_low = p_L_eq_P_low_L * ((1 - P_low_L_le_B_L) * L) + (1 - p_L_eq_P_low_L) * (P_low_L - B_L)/l_p_by_p_L_minus_P_low_L
        
        P_high_L_eq_p_L = (P_high_L == p_L).astype("float32")
        B_L_le_P_high_L = (B_L <= P_high_L).astype("float32")
        R_high = P_high_L_eq_p_L * ((1 - B_L_le_P_high_L) * L) + (1 - P_high_L_eq_p_L) * (B_L - P_high_L)/l_p_by_P_high_L_min_p_L
        
        R_0 = np.ceil(np.maximum(R_low, R_high))
        
        # one more bit 0
        B_L = L
        p_L_eq_P_low_L = (p_L == P_low_L).astype("float32")
        P_low_L_le_B_L = (P_low_L <= B_L).astype("float32")
        R_low = p_L_eq_P_low_L * ((1 - P_low_L_le_B_L) * L) + (1 - p_L_eq_P_low_L) * (P_low_L - B_L)/l_p/l_p_by_p_L_minus_P_low_L
        
        P_high_L_eq_p_L = (P_high_L == p_L).astype("float32")
        B_L_le_P_high_L = (B_L <= P_high_L).astype("float32")
        R_high = P_high_L_eq_p_L * ((1 - B_L_le_P_high_L) * L) + (1 - P_high_L_eq_p_L) * (B_L - P_high_L)/l_p_by_P_high_L_min_p_L
        
        R_L = np.ceil(np.maximum(R_low, R_high))
        
        R = np.minimum(R_0, R_L)
        
        R_by_l_p = R*l_p
        R_by_l_p_lt_max_stab_len = (R_by_l_p < max_stab_len).astype("float32")
        
        max_stab_len = R_by_l_p_lt_max_stab_len * np.maximum(R_by_l_p, 1)
        max_stab_R = R_by_l_p_lt_max_stab_len * R
        max_stab_l_p = R_by_l_p_lt_max_stab_len * l_p

    return max_stab_len.astype("float32"), max_stab_R.astype("float32"), max_stab_l_p.astype("float32")


class NormStability(torch.nn.Module):
    """
    calculate the normalized value-independent stability, which is standard stability over maximum stability.
    normalized stability is acutual stability/max possible stability
    All inputs should be on CPU
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
        self.len.data = self.stability.len.detach().clone()
    
    def forward(self):
        # parallel execution based on numpy speeds up by 30X
        assert self.len != 0, "Input bit stream length can't be 0."
        L = torch.pow(2, torch.ceil(torch.log2(self.len)))
        # use ceil for lower to avoid 0
        P_low_L_all = torch.floor(self.min_prob*L).clamp(0, L.item())
        # use ceil for upper to avoid the case that upper is smaller than lower, when bit stream length is small
        P_high_L_all = torch.ceil(self.max_prob*L).clamp(0, L.item())
        seach_range = (self.threshold * 2 * L + 1).type(torch.int32)

        max_stab_len, max_stab_R, max_stab_l_p = search_best_stab_parallel_numpy(P_low_L_all.type(torch.int32).numpy(), 
                                                                                 P_high_L_all.type(torch.int32).numpy(), 
                                                                                 L.type(torch.int32).numpy(), 
                                                                                 seach_range.numpy())
        
        self.max_stab.data = torch.from_numpy(np.maximum(1 - max_stab_len/self.len.numpy(), 0))
        self.max_stab_len.data = torch.from_numpy(max_stab_len)
        self.max_stab_l_p.data = torch.from_numpy(max_stab_l_p)
        self.max_stab_R.data = torch.from_numpy(max_stab_R)
        
        normstab = self.stability()/self.max_stab
        normstab[torch.isnan(normstab)] = 0
        # some normstab is larger than 1.0,
        # as the method based on the segmented uniform,
        # which is an approximation of the best case
        normstab.clamp_(0, 1)
        
        return normstab
    

def gen_ns_out_parallel_numpy(new_ns_len, out_cnt_ns, out_cnt_st, ns_gen, st_gen, bs_st, bs_ns, L):
    """
    This function is used to generate the output for NSbuilder
    All data are numpy arrays
    """
    out_cnt_ns_eq_new_ns_len = (out_cnt_ns >= new_ns_len).astype("int32")
    out_cnt_ns_eq_L = (out_cnt_ns >= L).astype("int32")
    
    ns_gen = 1 - out_cnt_ns_eq_new_ns_len
    st_gen = out_cnt_ns_eq_new_ns_len

    output = (ns_gen & bs_ns) | (st_gen & bs_st)
    
    out_cnt_ns = out_cnt_ns + ns_gen
    out_cnt_st = out_cnt_st + st_gen
    return out_cnt_ns, out_cnt_st, ns_gen, st_gen, output.astype("float32")


class NSbuilder(torch.nn.Module):
    """
    this module is the normalized stability builder. 
    it converts the normalized stability of a bitstream into the desired value.
    """
    def __init__(self,
                 bitwidth=8,
                 mode="bipolar",
                 normstability=0.5,
                 threshold=0.05,
                 value=None,
                 rng_dim=1,
                 rng="Sobol",
                 rtype=torch.float,
                 stype=torch.float):
        super(NSbuilder, self).__init__()
        
        self.bitwidth = bitwidth
        self.normstb = normstability
        self.val = value
        self.mode = mode
        self.val_shape = self.val.size()
        self.val_dim = len(self.val_shape)

        self.stype = stype
        self.rtype = rtype

        self.T = threshold
        self.L = torch.nn.Parameter(torch.tensor([2**self.bitwidth]).type(self.val.dtype), requires_grad=False)
        self.lp = torch.zeros_like(self.val)
        self.R = torch.ones_like(self.val)

        self.P_low = torch.zeros_like(self.val)
        self.P_up = torch.zeros_like(self.val)

        self.max_stable = torch.zeros_like(self.val)
        self.max_st_len = torch.zeros_like(self.val)
        self.new_st_len = torch.zeros_like(self.val)
        self.new_ns_len = torch.zeros_like(self.val)
        
        self.new_ns_val = torch.zeros_like(self.val)
        self.new_st_val = torch.zeros_like(self.val)
        self.new_ns_one = torch.zeros_like(self.val)
        self.new_st_one = torch.zeros_like(self.val)

        self.rng = RNG(
            bitwidth=bitwidth,
            dim=rng_dim,
            rng=rng,
            rtype=rtype)()

        self.ns_gen = torch.ones_like(self.val).type(torch.bool)
        self.st_gen = torch.zeros_like(self.val).type(torch.bool)
        
        self.out_cnt_ns = torch.zeros_like(self.val).type(torch.int32)
        self.out_cnt_st = torch.zeros_like(self.val).type(torch.int32)

        self.output = torch.zeros_like(self.val).type(stype)

        ## INIT:
        # Stage to calculate several essential params
        self.P_low = torch.max(self.val - self.T, torch.zeros_like(self.val))
        self.P_up = torch.min(torch.ones_like(self.val), self.val + self.T)
        upper = torch.min(torch.ceil(self.L * self.P_up), self.L)
        lower = torch.max(torch.floor(self.L * self.P_low), torch.zeros_like(self.L))
        
        seach_range = (self.T * 2 * self.L + 1).type(torch.int32)
        
        max_stab_len, max_stab_R, max_stab_l_p = search_best_stab_parallel_numpy(lower.type(torch.int32).numpy(), 
                                                                                 upper.type(torch.int32).numpy(), 
                                                                                 self.L.type(torch.int32).numpy(), 
                                                                                 seach_range.numpy())
        
        self.max_stable = torch.from_numpy(max_stab_len)
        self.lp = torch.from_numpy(max_stab_l_p)
        self.R = torch.from_numpy(max_stab_R)
        
        self.max_st_len = self.L - (self.max_stable)
        self.new_st_len = torch.ceil(self.max_st_len * self.normstb)
        self.new_ns_len = (self.L - self.new_st_len)
        
        val_gt_half = (self.val > 0.5).type(torch.float)
        self.new_ns_one = val_gt_half * (self.P_up * (self.new_ns_len + 1)) \
                        + (1 - val_gt_half) * torch.max((self.P_low * (self.new_ns_len + 1) - 1), torch.zeros_like(self.L))

        self.new_st_one = (self.val * self.L - self.new_ns_one)
        self.new_ns_val = self.new_ns_one / self.new_ns_len
        self.new_st_val = self.new_st_one / self.new_st_len

        self.src_st = SourceGen(self.new_st_val, self.bitwidth, self.mode, self.rtype)()
        self.src_ns = SourceGen(self.new_ns_val, self.bitwidth, self.mode, self.rtype)()
        self.bs_st = BSGen(self.src_st, self.rng)
        self.bs_ns = BSGen(self.src_ns, self.rng)

    def NSbuilder_forward(self):
        # parallel execution based on numpy speeds up by 90X
        ## Stage to generate output
        bs_st = self.bs_st(self.out_cnt_st).type(torch.int32).numpy()
        bs_ns = self.bs_ns(self.out_cnt_ns).type(torch.int32).numpy()
        out_cnt_ns, out_cnt_st, ns_gen, st_gen, output = gen_ns_out_parallel_numpy(self.new_ns_len.type(torch.int32).numpy(), 
                                                                                   self.out_cnt_ns.type(torch.int32).numpy(), 
                                                                                   self.out_cnt_st.type(torch.int32).numpy(), 
                                                                                   self.ns_gen.type(torch.int32).numpy(), 
                                                                                   self.st_gen.type(torch.int32).numpy(), 
                                                                                   bs_st, 
                                                                                   bs_ns, 
                                                                                   self.L.type(torch.int32).numpy())
        
        self.out_cnt_ns = torch.from_numpy(out_cnt_ns)
        self.out_cnt_st = torch.from_numpy(out_cnt_st)
        self.ns_gen = torch.from_numpy(ns_gen)
        self.st_gen = torch.from_numpy(st_gen)
        self.output = torch.from_numpy(output)
        
        return self.output.type(self.stype)

    def forward(self):
        return self.NSbuilder_forward()
    
