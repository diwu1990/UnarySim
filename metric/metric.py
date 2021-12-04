import torch

class Correlation(torch.nn.Module):
    """
    Calculate the stochastic cross correlation (SCC) between two input bit streams.
    If only one input bitstream is provided, it calculate the SCC between its delay and itself.
    Please refer to "Exploiting correlation in stochastic circuit design" for more details.
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
    
    
class ProgError(torch.nn.Module):
    """
    Calculate progressive error (pe) based on progressive precision of input bit stream.
    Progressive precision (pp): "Fast and accurate computation using stochastic circuits"
    "scale" is the scaling factor of the source data.
    """
    def __init__(
        self, 
        source, 
        hwcfg={
            "scale" : 1, 
            "mode" : "bipolar"
        }):
        super(ProgError, self).__init__()
        self.hwcfg = {}
        self.hwcfg["scale"] = hwcfg["scale"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()

        # in_value is always binary
        # after scaling, unipolar should be within (0, 1), bipolar should be within (-1, 1).
        # therefore, clamping with (-1, 1) always works
        self.scale = hwcfg["scale"]
        self.mode = hwcfg["mode"].lower()
        self.source = torch.clamp(source/self.scale, -1., 1.)
        assert self.mode == "unipolar" or "bipolar", \
            "Error: the hw config 'mode' in " + self + " class requires one of ['unipolar', 'bipolar']."
        self.cycle = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.one_cnt = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.pp = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.pe = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def Monitor(self, in_1):
        self.one_cnt.data = self.one_cnt.data.add(in_1.type(torch.float))
        self.cycle.data.add_(1)

    def forward(self):
        self.pp.data = self.one_cnt.div(self.cycle)
        if self.mode == "bipolar":
            self.pp.data = self.pp.mul(2).sub(1)
        self.pe.data = self.pp.sub(self.source)
        return self.pp, self.pe
    
    
class Stability(torch.nn.Module):
    """
    Calculate the stability of one bit stream. Please checkout following references for more information:
    1) "uGEMM: Unary Computing Architecture for GEMM Applications"
    2) "Normalized Stability: A Cross-Level Design Metric for Early Termination in Stochastic Computing"
    """
    def __init__(
        self, 
        source, 
        hwcfg={
            "scale" : 1, 
            "mode" : "bipolar", 
            "threshold" : 0.05
        }):
        super(Stability, self).__init__()
        self.hwcfg = {}
        self.hwcfg["scale"] = hwcfg["scale"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["threshold"] = hwcfg["threshold"]

        assert hwcfg["mode"].lower() == "unipolar" or "bipolar", \
            "Error: the hw config 'mode' in " + self + " class requires one of ['unipolar', 'bipolar']."

        self.source = source
        self.threshold = torch.nn.Parameter(torch.tensor([hwcfg["threshold"]]), requires_grad=False)
        self.cycle = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.pe = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.cycle_to_stable = torch.zeros_like(self.source) # cycle to reach (before) the stable state
        self.stability = torch.zeros_like(self.source)
        self.progerr = ProgError(self.source, hwcfg)
        
    def Monitor(self, in_1):
        self.progerr.Monitor(in_1)
        self.cycle = self.progerr.cycle
        _, self.pe = self.progerr()
        self.cycle_to_stable.add_(torch.gt(self.pe.abs(), self.threshold).type(torch.float).mul_(self.cycle - self.cycle_to_stable))
        
    def forward(self):
        self.stability = 1 - self.cycle_to_stable.clamp(1, self.cycle.item()).div(self.cycle)
        return self.stability

