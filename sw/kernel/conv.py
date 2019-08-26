import torch

class UnaryConv2d(torch.nn.modules.conv.Conv2d):
    """This is bipolar mul and non-scaled addition"""
    def __init__(self, in_channels, out_channels, kernel_size, output_shape,
                 binary_weight=torch.tensor([0]), binary_bias=torch.tensor([0]), bitwidth=8, 
                 stride=1, padding=0, dilation=1, 
                 groups=1, bias=True, padding_mode='zeros'):
        super(UnaryConv2d, self).__init__(in_channels, out_channels, kernel_size)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # data bit width
        self.buf_wght = binary_weight.clone().detach()
        if bias is True:
            self.buf_bias = binary_bias.clone().detach()
        self.bitwidth = bitwidth

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.groups = groups
        self.has_bias = bias
        self.padding_mode = padding_mode
        
        # random_sequence from sobol RNG
        self.rng = torch.quasirandom.SobolEngine(1).draw(pow(2,self.bitwidth)).view(pow(2,self.bitwidth))
        # convert to bipolar
        self.rng.mul_(2).sub_(1)
#         print(self.rng)

        # define the kernel linear
        self.kernel = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 
                              stride=self.stride, padding=self.padding, dilation=self.dilation, 
                              groups=self.groups, bias=self.has_bias, padding_mode=self.padding_mode)

        # define the RNG index tensor for weight
        self.rng_wght_idx = torch.zeros(self.kernel.weight.size(), dtype=torch.long)
        self.rng_wght = self.rng[self.rng_wght_idx]
        assert (self.buf_wght.size() == self.rng_wght.size()
               ), "Input binary weight size of 'kernel' is different from true weight."
        
        # define the RNG index tensor for bias if available, only one is required for accumulation
        if self.has_bias is True:
            print("Has bias.")
            self.rng_bias_idx = torch.zeros(self.kernel.bias.size(), dtype=torch.long)
            self.rng_bias = self.rng[self.rng_bias_idx]
            assert (self.buf_bias.size() == self.rng_bias.size()
                   ), "Input binary bias size of 'kernel' is different from true bias."

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # inverse
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # define the kernel_inverse, no bias required
        self.kernel_inv = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 
                              stride=self.stride, padding=self.padding, dilation=self.dilation, 
                              groups=self.groups, bias=False, padding_mode=self.padding_mode)
        
        # define the RNG index tensor for weight_inverse
        self.rng_wght_idx_inv = torch.zeros(self.kernel_inv.weight.size(), dtype=torch.long)
        self.rng_wght_inv = self.rng[self.rng_wght_idx_inv]
        assert (self.buf_wght.size() == self.rng_wght_inv.size()
               ), "Input binary weight size of 'kernel_inv' is different from true weight."
        
        self.in_accumulator = torch.zeros(output_shape)
        self.out_accumulator = torch.zeros(output_shape)
        self.output = torch.zeros(output_shape)
    
    def UnaryKernel_nonscaled_forward(self, input):
        # generate weight bits for current cycle
        self.rng_wght = self.rng[self.rng_wght_idx]
        self.kernel.weight.data = torch.gt(self.buf_wght, self.rng_wght).type(torch.float)
        print(self.rng_wght_idx.size())
        print(input.size())
        self.rng_wght_idx.add_(input.type(torch.long))
        if self.has_bias is True:
            self.rng_bias = self.rng[self.rng_bias_idx]
            self.kernel.bias.data = torch.gt(self.buf_bias, self.rng_bias).type(torch.float)
            self.rng_bias_idx.add_(1)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # inverse
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        self.rng_wght_inv = self.rng[self.rng_wght_idx_inv].type(torch.float)
        self.kernel_inv.weight.data = torch.le(self.buf_wght, self.rng_wght_inv).type(torch.float)
        self.rng_wght_idx_inv.add_(1).sub_(input.type(torch.long))
#         print(self.kernel(input).size())
        return self.kernel(input) + self.kernel_inv(1-input)
    
    def forward(self, input):
        self.in_accumulator.add_(self.UnaryKernel_nonscaled_forward(input))
#         .clamp_(-self.upper_bound, self.upper_bound)
        self.in_accumulator.sub_(self.offset)
        self.output = torch.gt(self.in_accumulator, self.out_accumulator).type(torch.float)
#         print("accumulator result:", self.in_accumulator, self.out_accumulator)
        self.out_accumulator.add_(self.output)
        return self.output
    