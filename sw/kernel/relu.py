import torch

class UnaryReLU(torch.nn.Module):
    """
    unary ReLU activation based on comparator
    data is always in bipolar representation
    the input bit streams are categorized into Sobol and Race like
    """
    def __init__(self, buf_dep=4, bitwidth=8, rng="Sobol"):
        super(UnaryReLU, self).__init__()
        if rng is "Sobol":
            self.buf_max = torch.nn.Parameter(torch.zeros(1).fill_(2**buf_dep - 1).type(torch.long), requires_grad=False)
            self.buf_half = torch.nn.Parameter(torch.zeros(1).fill_(2**(buf_dep - 1)).type(torch.long), requires_grad=False)
            self.acc = torch.nn.Parameter(torch.zeros(1).fill_(2**(buf_dep - 1)).type(torch.long), requires_grad=False)
        elif rng is "Race":
            self.threshold = torch.nn.Parameter(torch.zeros(1).fill_(2**(bitwidth - 1)).type(torch.long), requires_grad=False)
            self.acc = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
            self.cycle = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
        else:
            raise ValueError("UnaryReLU rng other than \"Sobol\" or \"Race\" is illegal.")
        self.rng = rng
    
    def UnaryReLU_forward_sobol(self, input):
        # check whether acc is larger than or equal to half.
        half_prob_flag = torch.ge(self.acc, self.buf_half).type(torch.int8)
        # only when input is 0 and flag is 1, output 0; otherwise 1
        output = input | (1 - half_prob_flag)
        # update the accumulator
        self.acc.data = self.acc.add(output.mul(2).sub(1).type(torch.long)).clamp(0, self.buf_max.item())
        return output
    
    def UnaryReLU_forward_race(self, input):
        # check reach half total cycle
        self.cycle.add_(1)
        half_cycle_flag = torch.gt(self.cycle, self.threshold).type(torch.int8)
        # check whether acc is larger than or equal to threshold, when half cycle is reached
        self.acc.data = self.acc.add(input.type(torch.long))
        half_prob_flag = torch.gt(self.acc, self.threshold).type(torch.int8)
        # if  1
        output = (1 - half_cycle_flag) * torch.ge(self.cycle, self.acc).type(torch.int8) + half_cycle_flag * half_prob_flag * input
        # update the accumulator
        return output

    def forward(self, input):
        if self.rng is "Sobol":
            return self.UnaryReLU_forward_sobol(input)
        elif self.rng is "Race":
            return self.UnaryReLU_forward_race(input)

