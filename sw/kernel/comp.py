import torch
import math

class UnaryCompare(torch.nn.modules.Module):
    """unary comparator"""
    def __init__(self, input_shape):
        super(UnaryCompare, self).__init__()

        self.input_shape = input_shape
        self.out_accumulator = torch.zeros(input_shape)
        self.out_acc_sign = torch.zeros(input_shape)
        self.output = torch.zeros(input_shape)

    def UnaryCompare_forward(self, input):
        self.out_acc_sign = torch.lt(self.out_accumulator, 0).type(torch.float)
        self.output = self.out_acc_sign + (1 - self.out_acc_sign) * input
        self.out_accumulator.add_(2 * self.output - 1)
        return self.output

    def forward(self, input):
        return self.UnaryCompare_forward(input)


