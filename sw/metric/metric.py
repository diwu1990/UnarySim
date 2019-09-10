import torch
import math

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

    def Monitor(self, in_1, in_2):
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
    """
    def __init__(self, in_value, mode="bipolar"):
        super(ProgressiveError, self).__init__()
        self.in_value = in_value
        self.mode = mode
        self.len = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.one_cnt = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.out_pp = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.err = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def Monitor(self, in_1):
        self.one_cnt.data = self.one_cnt.data.add(in_1.type(torch.float))
        self.len.data.add_(1)

    def forward(self):
        if self.mode is "unipolar" or self.mode is "bipolar":
            self.out_pp.data = self.one_cnt.div(self.len)
        else:
            raise ValueError("ProgressiveError mode is not implemented.")
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
        self.threshold = torch.nn.Parameter(torch.torch([threshold]), requires_grad=False)
        self.len = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.err = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.stable_len = torch.ones_like(in_value)
        self.stability = torch.zeros_like(in_value)
        self.pp = ProgressiveError(in_value, mode=mode)

    def Monitor(self, in_1):
        self.pp.Monitor(in_1)
        self.len = self.pp.len
        _, self.err = self.pp.Report()
        self.stable_len.add_(torch.gt(self.err.abs(), self.threshold).type(torch.float).mul_(self.len - self.stable_len))
        
    def forward(self):
        self.stability = 1 - self.stable_len.div(self.len)
        return self.stability
    
    
class NormStability(torch.nn.Module):
    """
    calculate the normalized value-independent stability, which is standard stability over maximum stability.
    normalized stability is acutual stability/max possible stability
    """
    def __init__(self, in_value, mode="bipolar", threshold=0.05):
        super(NormStability, self).__init__()
        self.in_value = in_value
        self.mode = mode
        self.threshold = threshold
        self.stability = Stability(in_value, mode=mode, threshold=threshold)
        if mode is "bipolar":
            self.min_prob = torch.max((in_value + 1) / 2 - threshold / 2, torch.tensor([0.]))
            self.max_prob = torch.min((in_value + 1) / 2 + threshold / 2, torch.tensor([1.]))
        else:
            self.min_prob = torch.max(in_value - threshold, torch.tensor([0.]))
            self.max_prob = torch.min(in_value + threshold, torch.tensor([1.]))
        self.len = 0.0
        self.in_shape = in_value.size()
        self.max_stab = torch.zeros(self.in_shape)

    def Monitor(self, in_1):
        self.stability.Monitor(in_1)
        self.len = self.stability.len
        
    def forward(self):
        assert self.len != 0, "Input bit stream length can't be 0."
        dim = len(self.in_shape)
        assert dim <= 4, "Input dimension larger than 4 is not implemented."
        length_gcd = pow(2, math.ceil(math.log2(self.len)))
        lower = torch.ceil(self.min_prob*length_gcd)
        upper = torch.ceil(self.max_prob*length_gcd)
        if dim == 1:
            for index_0 in range(self.in_shape[0]):
                max_stab_len = length_gcd
                for val in range(lower[index_0].type(torch.int), 
                                 upper[index_0].type(torch.int)):
                    val_gcd = math.gcd(val, length_gcd)
                    max_stab_len = min(max_stab_len, length_gcd/val_gcd)
                self.max_stab[index_0] = 1 - max_stab_len/length_gcd
                
        if dim == 2:
            for index_0 in range(self.in_shape[0]):
                for index_1 in range(self.in_shape[1]):
                    max_stab_len = length_gcd
                    for val in range(lower[index_0][index_1].type(torch.int), 
                                     upper[index_0][index_1].type(torch.int)):
                        val_gcd = math.gcd(val, length_gcd)
                        max_stab_len = min(max_stab_len, length_gcd/val_gcd)
                    self.max_stab[index_0][index_1] = 1 - max_stab_len/length_gcd
                    
        if dim == 3:
            for index_0 in range(self.in_shape[0]):
                for index_1 in range(self.in_shape[1]):
                    for index_2 in range(self.in_shape[2]):
                        max_stab_len = length_gcd
                        for val in range(lower[index_0][index_1][index_2].type(torch.int), 
                                         upper[index_0][index_1][index_2].type(torch.int)):
                            val_gcd = math.gcd(val, length_gcd)
                            max_stab_len = min(max_stab_len, length_gcd/val_gcd)
                        self.max_stab[index_0][index_1][index_2] = 1 - max_stab_len/length_gcd
                        
        if dim == 4:
            for index_0 in range(self.in_shape[0]):
                for index_1 in range(self.in_shape[1]):
                    for index_2 in range(self.in_shape[2]):
                        for index_3 in range(self.in_shape[3]):
                            max_stab_len = length_gcd
                            for val in range(lower[index_0][index_1][index_2][index_3].type(torch.int), 
                                             upper[index_0][index_1][index_2][index_3].type(torch.int)):
                                val_gcd = math.gcd(val, length_gcd)
                                max_stab_len = min(max_stab_len, length_gcd/val_gcd)
                            self.max_stab[index_0][index_1][index_2][index_3] = 1 - max_stab_len/length_gcd
                            
        return self.stability.Report()/self.max_stab
    