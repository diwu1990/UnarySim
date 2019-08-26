import torch
import math

class Correlation():
    """
    calculate the correlation between two input bit streams.
    """
    def __init__(self, in_shape):
        super(Correlation, self).__init__()
        self.in_shape = in_shape
        self.paired_00_d = torch.zeros(in_shape)
        self.paired_01_c = torch.zeros(in_shape)
        self.paired_10_b = torch.zeros(in_shape)
        self.paired_11_a = torch.zeros(in_shape)
        self.len = 0.0

    def Monitor(self, in_1, in_2):
        self.paired_00_d.add_(torch.eq(in_1, 0).type(torch.float)*torch.eq(in_2, 0).type(torch.float))
        self.paired_01_c.add_(torch.eq(in_1, 0).type(torch.float)*torch.eq(in_2, 1).type(torch.float))
        self.paired_10_b.add_(torch.eq(in_1, 1).type(torch.float)*torch.eq(in_2, 0).type(torch.float))
        self.paired_11_a.add_(torch.eq(in_1, 1).type(torch.float)*torch.eq(in_2, 1).type(torch.float))
        self.len += 1
    
    def Report(self):
        ad_minus_bc = self.paired_11_a * self.paired_00_d - self.paired_10_b * self.paired_01_c
        ad_gt_bc = torch.gt(ad_minus_bc, 0).type(torch.float)
        ad_lt_bc = torch.lt(ad_minus_bc, 0).type(torch.float)
        a_plus_b = self.paired_11_a + self.paired_10_b
        a_plus_c = self.paired_11_a + self.paired_01_c
        a_minus_d = self.paired_11_a - self.paired_00_d
        corr_ad_gt_bc = ad_minus_bc.div(
            torch.max(
                torch.min(a_plus_b, a_plus_c).mul(self.len).sub(a_plus_b.mul(a_plus_c)), 
                torch.ones(self.in_shape)
            )
        )
        corr_ad_lt_bc = ad_minus_bc.div(
            torch.max(
                a_plus_b.mul(a_plus_c).sub(torch.max(a_minus_d, torch.zeros(self.in_shape)).mul(self.len)),
                torch.ones(self.in_shape)
            )
        )
        return ad_gt_bc * corr_ad_gt_bc + ad_lt_bc * corr_ad_lt_bc
    
    
class ProgressivePrecision(object):
    """
    calculate progressive precision of input bit stream.
    """
    def __init__(self, in_value, mode="unipolar"):
        super(ProgressivePrecision, self).__init__()
        self.in_value = in_value
        self.mode = mode
        self.len = 0.0
        self.one_cnt = 0.0
        self.out_pp = 0.0
        self.err = 0.0
    
    def Monitor(self, in_1):
        self.one_cnt += in_1.type(torch.float)
        self.len += 1

    def Report(self):
        if self.mode is "unipolar" or self.mode is "bipolar":
            self.out_pp = self.one_cnt / self.len
        else:
            raise ValueError("ProgressivePrecision mode is not implemented.")
        if self.mode is "bipolar":
            self.out_pp = 2 * self.out_pp - 1
        self.err = self.out_pp - self.in_value
        return self.err
    
    
class Stability():
    """
    calculate the stability of one bit stream.
    """
    def __init__(self, in_value, mode="bipolar", threshold=0.05):
        super(Stability, self).__init__()
        self.in_value = in_value
        self.mode = mode
        self.threshold = threshold
        self.len = 0.0
        self.err = 0.0
        self.stable_len = torch.zeros(in_value.size())
        self.stability = torch.zeros(in_value.size())
        self.pp = ProgressivePrecision(in_value, mode=mode)

    def Monitor(self, in_1):
        self.pp.Monitor(in_1)
        self.len = self.pp.len
        self.err = self.pp.Report()
        update = torch.gt(self.err.abs(), self.threshold).type(torch.float)
        self.stable_len.add_(update.mul(self.len - self.stable_len))
        
    def Report(self):
        self.stability = 1 - self.stable_len.div(self.len)
        return self.stability
    
    
class NormStability():
    """
    calculate the normalized value-independent stability, which is standard stability over maximum stability.
    maximum stability is (1-min{(GCD(X, L) for X in (max(0,p-Pthd)*L, min(1,p+P_thd)*L)}/L)
    """
    def __init__(self, in_value, mode="bipolar", threshold=0.05):
        super(NormStability, self).__init__()
        self.in_value = in_value
        self.mode = mode
        self.threshold = threshold
        self.stability = Stability(in_value, mode=mode, threshold=threshold)
        self.min_prob = torch.max(in_value - threshold, torch.tensor([0.]))
        self.max_prob = torch.min(in_value + threshold, torch.tensor([1.]))
        self.len = 0.0
        self.in_shape = in_value.size()
        self.max_stab = torch.zeros(self.in_shape)

    def Monitor(self, in_1):
        self.stability.Monitor(in_1)
        self.len = self.stability.len
        
    def Report(self):
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
    