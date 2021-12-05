import torch

class FSUHardtanh(torch.nn.Identity):
    """
    This module is used for inference in unary domain.
    """
    def __init__(self):
        super(FSUHardtanh, self).__init__()


class HUBHardtanh(torch.nn.Hardtanh):
    """
    Inputs within range [-1, +1] directly pass through, while inputs outsides will be clipped to -1 and +1.
    This module is used for training and inference in binary domain.
    """
    def __init__(self):
        super(HUBHardtanh, self).__init__()
    
