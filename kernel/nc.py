import torch

# Inherit from Function
class NCFireStep(torch.autograd.Function):
    """
    Using step function for neuron firing with approximate gradients and without surrogating gradients.
    """
    @staticmethod
    def forward(ctx, mem, vth, widthg):
        ctx.save_for_backward(mem)
        ctx.mem = mem
        ctx.vth = vth
        ctx.widthg = widthg
        return (mem > vth).type(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        mem, = ctx.saved_tensors
        diff_abs_legal = torch.abs(mem - ctx.vth) < ctx.widthg
        grad_input = grad_output*torch.where(diff_abs_legal, 1./(2*ctx.widthg), 0.)

        return grad_input, None, None

