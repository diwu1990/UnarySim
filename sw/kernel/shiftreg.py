import torch

class ShiftReg(torch.nn.Module):
    """
    this module is a shift register for int8 tensor.
    it takes in unary bits and output shifter register value, as well as the total number of 1s in current shift register.
    """
    def __init__(self,
                 depth=8,
                 bstype=torch.float):
        super(ShiftReg, self).__init__()
        self.depth = 8
        self.sr = torch.nn.Parameter(torch.tensor([x%2 for x in range(0, depth)]).type(bstype), requires_grad=False)
        self.init = torch.nn.Parameter(torch.ones(1).type(torch.bool), requires_grad=False)
        self.bstype = bstype

    def ShiftReg_forward(self, input, mask=None, index=0):
        if self.init.item() is True:
            new_shape = [1 for _ in range(len(input.shape))]
            new_shape.insert(0, self.depth)
            self.sr.data = input.repeat(new_shape)
            for i in range(self.depth):
                self.sr[i].fill_(i%2)
            self.init.data.fill_(False)
        
        # do shifting
        # output
        out = self.sr[index]
        # sum in current shift register
        cnt = torch.sum(self.sr, 0)
        if mask is None:
            self.sr.data = torch.roll(self.sr, -1, 0)
            self.sr.data[self.depth-1] = input.clone().detach()
        else:
            assert mask.size() == input.size(), "size of the enable mask unmatches that of input"
            mask_val = mask.type(self.bstype)
            sr_shift = torch.roll(self.sr, -1, 0)
            sr_shift[self.depth-1] = input.clone().detach()
            sr_no_shift = self.sr.clone().detach()
            self.sr.data = mask_val * sr_shift + (1 - mask_val) * sr_no_shift
        return out, cnt

    def forward(self, input, mask=None, index=0):
        return self.ShiftReg_forward(input, mask=mask, index=index)

    