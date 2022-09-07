import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone1, backbone2, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.classifier = classifier
        p = 8
        # self.conv1 = conv1x1(256, 32)
        # self.conv2 = conv1x1(512, 64)
        # self.conv3 = conv1x1(1024, 128)
        # self.conv4 = conv1x1(2048, 256)

        self.conv1 = conv1x1(256, 256//p)
        self.conv2 = conv1x1(512, 512//p)
        self.conv3 = conv1x1(1024, 1024//p)
        self.conv4 = conv1x1(2048, 2048//p)

        # self.sc1 = CAM_Module(288)  # 通道注意力模块
        # self.sc2 = CAM_Module(576)  # 通道注意力模块
        # self.sc3 = CAM_Module(1152)  # 通道注意力模块
        # self.sc4 = CAM_Module(2304)  # 通道注意力模块

        # from torchsummary import summary
        # print(backbone1)
        # summary(model, input_size=[(1, 590, 1006)], batch_size=2, device="cpu")


    def forward(self, x0, x1,gini=None):  # x0.size=(batch,1,590,1006)   x1也是
        input_shape = x0.shape[-2:]

        features1 = self.backbone1(x0, gini)  # out是2048个feature_map；low_level是256个feature_map
        features2 = self.backbone2(x1, gini)
        feature = OrderedDict()
        # print("out:",features1['out'].size())
        # print("low_level1:",features1['low_level1'].size())
        # print("low_level2:",features1['low_level2'].size())
        # print("low_level3:",features1['low_level3'].size())

        features2['out'] = self.conv4(features2['out'])
        features2['low_level1'] = self.conv1(features2['low_level1'])
        features2['low_level2'] = self.conv2(features2['low_level2'])
        features2['low_level3'] = self.conv3(features2['low_level3'])

        feature['out'] = torch.cat((features1['out'], features2['out']), dim=1)
        feature['low_level1'] = torch.cat((features1['low_level1'], features2['low_level1']), dim=1)
        feature['low_level2'] = torch.cat((features1['low_level2'], features2['low_level2']), dim=1)
        feature['low_level3'] = torch.cat((features1['low_level3'], features2['low_level3']), dim=1)

        #加通道注意力
        # feature['out'] = self.sc4(feature['out'])
        # feature['low_level1'] = self.sc1(feature['low_level1'])
        # feature['low_level2'] = self.sc2(feature['low_level2'])
        # feature['low_level3'] = self.sc3(feature['low_level3'])

        x = self.classifier(feature, gini)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x, gini):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if gini is not None and name in ['layer1', 'layer2', 'layer3', 'layer4']:
                _, c, w, h = x.size()
                gini_x = F.interpolate(gini, size=(w, h), mode='bilinear', align_corners=False)
                gini_x = torch.repeat_interleave(gini_x, repeats=c, dim=1)
                # print("gini：", name, gini_x.size(), x.size())
                x = x + x * gini_x
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))  # β尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.softmax = nn.Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        m_batchsize, C, height, width = x.size()
        # A -> (N,C,HW)
        proj_query = x.view(m_batchsize, C, -1)
        # A -> (N,HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # 矩阵乘积，通道注意图：X -> (N,C,C)
        energy = torch.bmm(proj_query, proj_key)
        # 这里实现了softmax用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy_new)
        # A -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        # XA -> （N,C,HW）
        out = torch.bmm(attention, proj_value)
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


# class PAM_Module(nn.Module):
#     """ Position attention module"""
#
#     # Ref from SAGAN
#     def __init__(self, in_dim):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重
#
#         self.softmax = nn.Softmax(dim=-1)  # 对每一行进行softmax
#
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B × C × H × W)
#             returns :
#                 out : attention value + input feature
#                 attention: B × (H×W) × (H×W)
#         """
#         m_batchsize, C, height, width = x.size()
#         # B -> (N,C,HW) -> (N,HW,C)
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
#         # C -> (N,C,HW)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
#         # BC，空间注意图 -> (N,HW,HW)
#         energy = torch.bmm(proj_query, proj_key)
#         # S = softmax(BC) -> (N,HW,HW)
#         attention = self.softmax(energy)
#         # D -> (N,C,HW)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
#         # DS -> (N,C,HW)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # torch.bmm表示批次矩阵乘法
#         # output -> (N,C,H,W)
#         out = out.view(m_batchsize, C, height, width)
#
#         out = self.gamma * out + x
#         return out
