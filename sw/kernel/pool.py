import torch
import math
class UnaryAvgPool2d(torch.nn.modules.pooling.AvgPool2d):
    """unary 2d average pooling based on scaled addition"""
    def __init__(self, kernel_size, input_shape, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(UnaryAvgPool2d, self).__init__(kernel_size, input_shape)

        self.input_shape = input_shape
        
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        
        self.output_shape = list(input_shape)
        # data bit width
        if stride is None:
            if isinstance(kernel_size, int):
                self.scale = kernel_size*kernel_size
                self.output_shape[2] = int((input_shape[2] + 2 * padding - kernel_size) / kernel_size + 1)
                self.output_shape[3] = int((input_shape[3] + 2 * padding - kernel_size) / kernel_size + 1)
            elif isinstance(kernel_size, tuple):
                self.scale = kernel_size[0]*kernel_size[1]
                self.output_shape[2] = int((input_shape[2] + 2 * padding - kernel_size[0]) / stride[0] + 1)
                self.output_shape[3] = int((input_shape[3] + 2 * padding - kernel_size[1]) / stride[1] + 1)
        else:
            # to do
            pass
        
        self.in_accumulator = torch.zeros(self.output_shape)
        self.output = torch.zeros(self.output_shape)
        
        # define the kernel avgpool2d
        self.avgpool2d = torch.nn.AvgPool2d(self.kernel_size, 
                                            stride=self.stride, padding=self.padding, 
                                            ceil_mode=self.ceil_mode, 
                                            count_include_pad=self.count_include_pad, 
                                            divisor_override=self.divisor_override)
        
    def UnaryScaledADD_forward(self, input):
        self.in_accumulator.add_(self.avgpool2d(input))
        self.output = torch.ge(self.in_accumulator, 1).type(torch.float)
        self.in_accumulator.sub_(self.output)
        return self.output

    def forward(self, input):
        return self.UnaryScaledADD_forward(input)


