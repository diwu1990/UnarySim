import torch

class ShiftReg(torch.nn.Module):
    """
    This module is a shift register.
    Its input is bitstreams.
    Its output are shifter register value and the one count in the shift register.
    """
    def __init__(
        self,
        hwcfg={
            "entry" : 8
        }, 
        swcfg={
            "stype" : torch.float
        }):
        super(ShiftReg, self).__init__()
        self.hwcfg = {}
        self.hwcfg["entry"] = hwcfg["entry"]

        self.swcfg = {}
        self.swcfg["stype"] = swcfg["stype"]

        self.entry = hwcfg["entry"]
        self.stype = swcfg["stype"]
        self.sr = torch.nn.Parameter(torch.tensor([x%2 for x in range(0, self.entry)]).type(self.stype), requires_grad=False)
        self.first = True

    def ShiftReg_forward(self, input, mask=None, index=0):
        if self.first is True:
            new_shape = [1 for _ in range(len(input.shape))]
            new_shape.insert(0, self.entry)
            self.sr.data = input.repeat(new_shape)
            for i in range(self.entry):
                self.sr[i].fill_(i%2)
            self.first = False
        
        # do shifting
        # output
        out = self.sr[index]
        # sum in current shift register
        cnt = torch.sum(self.sr, 0)
        if mask is None:
            self.sr.data = torch.roll(self.sr, -1, 0)
            self.sr.data[self.entry-1] = input.clone().detach()
        else:
            assert mask.size() == input.size(), "Error: size of the enable mask unmatches that of input in " + str(self) + " class."
            mask_val = mask.type(self.stype)
            sr_shift = torch.roll(self.sr, -1, 0)
            sr_shift[self.entry-1] = input.clone().detach()
            sr_no_shift = self.sr.clone().detach()
            self.sr.data = mask_val * sr_shift + (1 - mask_val) * sr_no_shift
        return out, cnt

    def forward(self, input, mask=None, index=0):
        return self.ShiftReg_forward(input, mask=mask, index=index)

    