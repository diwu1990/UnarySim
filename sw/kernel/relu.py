import torch

class UnaryReLU(torch.nn.Module):
    """
    unary ReLU activation based on comparator
    data is always in bipolar mode
    """
    def __init__(self, buf_dep=4, mode="bipolar", compensation=None):
        super(UnaryReLU, self).__init__()
        if mode is not "bipolar":
            raise ValueError("UnaryReLU mode other than \"bipolar\" is illegal.")
        
        self.buf_max = torch.nn.Parameter(torch.zeros(1).fill_(2**buf_dep - 1).type(torch.uint8), requires_grad=False)
        self.buf_half = torch.nn.Parameter(torch.zeros(1).fill_(2**(buf_dep - 1)).type(torch.uint8), requires_grad=False)
        self.acc = torch.nn.Parameter(torch.zeros(1).fill_(2**(buf_dep - 1)).type(torch.uint8), requires_grad=False)

    def UnaryReLU_forward(self, input):
        # when accumulator result is no less than half, output bit stream is original.
        # when accumulator result is less than half, output bit stream is half 0 and half 1. (some designs use smooth fluctuation instead)
        output = 1 - torch.ge(self.acc, self.buf_half).type(torch.uint8) * (1 - input)
        self.acc.data = self.acc.add(input.mul(2).sub(1)).clamp(0, self.buf_max.item())
        return output

    def forward(self, input):
        return self.UnaryReLU_forward(input)

