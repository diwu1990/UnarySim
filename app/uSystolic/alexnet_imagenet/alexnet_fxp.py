import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from UnarySim.kernel.conv import FxpConv2d
from UnarySim.kernel.linear import FxpLinear
from UnarySim.kernel.utils import conv2d_output_shape


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, pretrained_model_state_dict=None, bitwidth=None, keep_res="input", more_res="input"):
        super(AlexNet, self).__init__()
        if pretrained_model_state_dict is None:
            self.features = nn.Sequential(
                FxpConv2d(3, 64, kernel_size=11, stride=4, padding=2, keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                FxpConv2d(64, 192, kernel_size=5, padding=2, keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                FxpConv2d(192, 384, kernel_size=3, padding=1, keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                FxpConv2d(384, 256, kernel_size=3, padding=1, keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                FxpConv2d(256, 256, kernel_size=3, padding=1, keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                FxpLinear(256 * 6 * 6, 4096, keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                FxpLinear(4096, 4096, keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                FxpLinear(4096, num_classes, keep_res=keep_res, more_res=more_res),
            )
        else:
            param_list = [param for param in pretrained_model_state_dict]
            print("load model parameters: ", param_list)
            state_list = [pretrained_model_state_dict[param] for param in param_list]
#             output_size_list = []
#             output_size_list[0] = 
            
            self.features = nn.Sequential(
                FxpConv2d(3, 64, kernel_size=11, stride=4, padding=2, binary_weight=state_list[0], binary_bias=state_list[1], bitwidth=bitwidth[0], keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                FxpConv2d(64, 192, kernel_size=5, padding=2, binary_weight=state_list[2], binary_bias=state_list[3], bitwidth=bitwidth[1], keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                FxpConv2d(192, 384, kernel_size=3, padding=1, binary_weight=state_list[4], binary_bias=state_list[5], bitwidth=bitwidth[2], keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                FxpConv2d(384, 256, kernel_size=3, padding=1, binary_weight=state_list[6], binary_bias=state_list[7], bitwidth=bitwidth[3], keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                FxpConv2d(256, 256, kernel_size=3, padding=1, binary_weight=state_list[8], binary_bias=state_list[9], bitwidth=bitwidth[4], keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                FxpLinear(256 * 6 * 6, 4096, binary_weight=state_list[10], binary_bias=state_list[11], bitwidth=bitwidth[5], keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                FxpLinear(4096, 4096, binary_weight=state_list[12], binary_bias=state_list[13], bitwidth=bitwidth[6], keep_res=keep_res, more_res=more_res),
                nn.ReLU(inplace=True),
                FxpLinear(4096, num_classes, binary_weight=state_list[14], binary_bias=state_list[15], bitwidth=bitwidth[7], keep_res=keep_res, more_res=more_res),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
    else:
        state_dict = None
    model = AlexNet(pretrained_model_state_dict=state_dict, **kwargs)
    return model
