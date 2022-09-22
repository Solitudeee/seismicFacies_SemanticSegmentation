import torch
from torch import nn, einsum
from torch.nn import functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from utils import _SimpleSegmentationModel
from einops import rearrange

__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(2048, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()


    def forward(self, features1):
        low_level_feature = self.project(features1['low_level1'])
        output_feature = features1['out']
        output_feature = self.aspp(output_feature)

        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))



    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


class CondGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, label_nc, kernel_size, stride=1, padding=0, dilation=1,
                 pad_type='zero',
                 activation='tanh', norm='bn', sn=False):
        super(CondGatedConv2d, self).__init__()

        self.out_channels = out_channels

        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(2304)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(2304)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(2304)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
            # self.mask_conv2d = spectral_norm(
            # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            # self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
        self.sigmoid = torch.nn.Sigmoid()

        ####### mod 1 ########
        # nhidden = out_channels // 2
        # nhidden = 128
        nhidden = 1024
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(in_channels, nhidden, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)

        ####### mod 2 ########
        self.mlp_shared_2 = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.mlp_shared_3 = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.mlp_gamma_ctx_gamma = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta_ctx_gamma = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)

        self.mlp_gamma_ctx_beta = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta_ctx_beta = nn.Conv2d(nhidden, out_channels, kernel_size=3, padding=1)

        self.alpha1 = nn.Parameter(torch.zeros(2, 256, 72, 124))
        self.alpha2 = nn.Parameter(torch.zeros(2, 256, 72, 124))
        self.project1 = nn.Sequential(
            nn.Conv2d(256, 2304, 1, bias=False),
            nn.BatchNorm2d(2304),
            nn.ReLU(inplace=True),
        )


    def forward(self, x, gini):
        # print(x.size())  #torch.Size([2, 2304, 74, 126])
        conv = self.conv2d(x)   #torch.Size([2, 256, 72, 124])
        norm = self.norm(x)    #torch.Size([2, 2304, 74, 126])

        ####### mod 2 ########
        gini = F.interpolate(gini, size=conv.size()[2:], mode='nearest')
        gini_F = 1-gini

        ctx1 = self.mlp_shared_2(gini)
        ctx2 = self.mlp_shared_3(gini_F)
        gamma_ctx_gamma = self.mlp_gamma_ctx_gamma(ctx1)
        beta_ctx_gamma = self.mlp_beta_ctx_gamma(ctx1)
        gamma_ctx_beta = self.mlp_gamma_ctx_beta(ctx2)
        beta_ctx_beta = self.mlp_beta_ctx_beta(ctx2)


        ####### mod 1 ########
        # x_conv = self.conv_x(x)
        actv = self.mlp_shared(x)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        gamma = F.interpolate(gamma, size=gamma_ctx_gamma.size()[2:], mode='nearest')
        beta = F.interpolate(beta, size=gamma_ctx_beta.size()[2:], mode='nearest')
        gamma = gamma * (1. + gamma_ctx_gamma) + beta_ctx_gamma
        beta = beta * (1. + gamma_ctx_beta) + beta_ctx_beta

        out_norm = conv * (1. + gamma+self.alpha1) + beta+self.alpha2

        out_norm = self.activation(out_norm)
        gate = self.project1(out_norm)
        gate = 1. + torch.tanh(gate)

        gate = F.interpolate(gate, size=norm.size()[2:], mode='nearest')

        return norm*gate
