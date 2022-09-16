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


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重

        self.softmax = nn.Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × (H×W) × (H×W)
        """
        m_batchsize, C, height, width = x.size()
        # B -> (N,C,HW) -> (N,HW,C)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # C -> (N,C,HW)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # BC，空间注意图 -> (N,HW,HW)
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(BC) -> (N,HW,HW)
        attention = self.softmax(energy)
        # D -> (N,C,HW)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # DS -> (N,C,HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # torch.bmm表示批次矩阵乘法
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class Attention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=2,
            dim_head=128,
            rel_pos_emb=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(256, inner_dim * 3, 1, bias=False)

        self.to_k2 = nn.Conv2d(2048, inner_dim, 1, bias=False)
        self.to_k3 = nn.Conv2d(48, inner_dim, 1, bias=False)
        # rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        # self.pos_emb = rel_pos_class(fmap_size, dim_head)
        self.cov1x1 = conv1x1(256*3,256)
        self.alpha1 = nn.Parameter(torch.ones(2, 2, 1, 1)) #B C H W
        self.alpha2 = nn.Parameter(torch.ones(2, 2, 1, 1))

    def forward(self, fmap,x):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k1, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k1, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (q, k1, v))

        k2 = self.to_k2(x)
        k2 = rearrange(k2, 'b (h d) x y -> b h (x y) d', h=heads)
        # k2 = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (q2, k2))

        # k3 = self.to_k3(y)
        # k3 = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (k3,))
        # k3 = rearrange(k3, 'b (h d) x y -> b h (x y) d', h=heads)
        # q3, k3 = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (q3, k3))

        q = q * self.scale

        sim1 = einsum('b h i d, b h j d -> b h i j', q, k1)
        sim2 = einsum('b h i d, b h j d -> b h i j', q, k2)
        # sim3 = einsum('b h i d, b h j d -> b h i j', q, k3)

        sim = sim1 + sim2


        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return out

class CAM_Module_my(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module_my, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))  # β尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.softmax = nn.Softmax(dim=-1)  # 对每一行进行softmax
        self.bate = nn.Parameter(torch.zeros(1))  # β尺度系数初始化为0，并逐渐地学习分配到更大的权重

    def forward(self, x,y):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        #-----------------X-----------
        m_batchsize, C, height, width = x.size()
        # A -> (N,C,HW)
        proj_query = x.view(m_batchsize, C, -1)
        print(proj_query.size())
        # A -> (N,HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        print(proj_key.size())
        # 矩阵乘积，通道注意图：X -> (N,C,C)
        energy = torch.bmm(proj_query, proj_key)
        print(energy.size())
        # 这里实现了softmax用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        print(energy_new.size())
        # -----------------Y-----------
        m_batchsize_y, C_y, height_y, width_y = y.size()
        # A -> (N,C,HW)
        proj_query_y = y.view(m_batchsize_y, C_y, -1)
        # A -> (N,HW,C)
        proj_key_y = y.view(m_batchsize_y, C_y, -1).permute(0, 2, 1)
        # 矩阵乘积，通道注意图：X -> (N,C,C)
        energy_y = torch.bmm(proj_query_y, proj_key_y)
        # 这里实现了softmax用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy_new_y = torch.max(energy_y, -1, keepdim=True)[0].expand_as(energy_y) - energy_y
        # -----------------合并-----------
        # attention = self.softmax(energy_new+self.bate*energy_new_y)
        attention = self.softmax(energy_new)
        print(attention.size())
        # A -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        print(proj_value.size())
        # XA -> （N,C,HW）
        out = torch.bmm(attention, proj_value)
        print(out.size())
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)
        print(out.size())

        out = self.gamma * out + x
        return out



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

        self.project1 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.project2 = nn.Sequential(
            nn.Conv2d(2304, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )


        self.aspp = ASPP(2304, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()
        self.at = Attention(dim=256)
        self.cam = CAM_Module(in_dim=2304)
        self.cgc = CondGatedConv2d(in_channels=2304,out_channels=256,label_nc=1,kernel_size=3)


    def forward(self, features1, features2, gini=None):
        low_level_feature = self.project(features1['low_level1'])
        output_feature = self.cam(torch.cat([features1['out'], self.project1(features2['out'])],dim=1))

        output_feature = self.cgc(output_feature,gini)
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
                 activation='elu', norm='bn', sn=False):
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
        # if self.activation:
        #     out = self.activation(out_norm)
        out_norm = self.project1(out_norm)
        gate = 1. + torch.tanh(out_norm)

        gate = F.interpolate(gate, size=norm.size()[2:], mode='nearest')

        return norm*gate
