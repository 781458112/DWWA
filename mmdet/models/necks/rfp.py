# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, xavier_init
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from ..builder import NECKS, build_backbone
from .fpn import FPN
from ..backbones.xiaobo_resnet_s import softmaxattention1 as  softmaxattention1_
class softmaxattention1(softmaxattention1_):
    def __init__(self,channel,dcn):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(softmaxattention1, self).__init__(channel,dcn)
        self.cv1 = build_conv_layer(None, channel, channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.cv2 = build_conv_layer(None, channel, channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.cv3 = build_conv_layer(None, channel, channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        # self.cv4 = build_conv_layer(dcn, 4096, 4096, kernel_size=1, stride=2, padding=0, dilation=1, bias=False)
        self.cv5 = build_conv_layer(None, channel*2, channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # print('x1',x.shape)
        y1 = self.cv1(x)  ##[B,C,H,W]  C
        # print('y1', y1.shape)
        y2 = self.cv2(x)  ##  B
        y1se = self.se1(y1)
        y2se = self.se2(y2)
        y3 = self.cv3(x)
        # print('y2', y2.shape)##[B,C,H,W]
        # x_= x.view([x.size()[0], x.size()[1], -1])#[B,C,H*w]
        # print('x_', x_.shape)  ##[B,C,H,W]

        y2_ = y2se.view([y2.size()[0], y2.size()[1], -1])  # [B,C,H*w] 0 1 2
        # print('y2_', y2_.shape)  ##[B,C,H*w] 0 1 2
        y1_ = y1se.view([y1.size()[0], y1.size()[1], -1])  # [B,C,H*w] 0 1 2
        # print('y1_ ', y1_ .shape)  ##[B,C,H*w] 0 1 2

        y2_T = y2_.permute([0, 2, 1])  # [B,H*w,C]  0 2 1  D   ###K
        # print('y2_T ', y2_T.shape)# [B,H*w,C]
        # c=y1_*y2_T  #[B,C,H*w]  * [B,H*w,C]
        c = torch.matmul(y2_T, y1_)  # [B,H*w,H*w]  ###QK
        # print('c ', c.shape)  # [B,H*w,H*w]
        C_weight = F.softmax(c, dim=-1)  ##[B,H*w,H*w]  E
        # print('C_weight', C_weight.shape)  # [B,H*w,H*w]
        # y=C*x
        y3_ = y3.view([x.size()[0], x.size()[1], -1])
        y_last = torch.matmul(y3_, C_weight)
        y_last = torch.nn.functional.normalize(y_last)
        # y_last = torch.matmul(x_,C_weight) #[B,C,H*w] *[B,H*w,H*w]  F-  ###yuanshi
        y_last_ = y_last.view([x.size()[0], x.size()[1], x.size()[2], x.size()[3]])  # [B,C,H*w] 0 1 2
        # y_last_ = torch.nn.functional.normalize(y_last_)
        zhao=torch.add(y_last_, x)
        # zhao=torch.cat((y_last_,x),dim=1)
        # zhao_new = self.cv5(zhao)
        return zhao


class ASPP(BaseModule):
    """ASPP (Atrous Spatial Pyramid Pooling)

    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations=(1, 3, 6, 1),
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super().__init__(init_cfg)
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            # conv =build_conv_layer(dict(type='DCN', deform_groups=1),in_channels,out_channels,kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation,bias=False)
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


@NECKS.register_module()
class RFP(FPN):
    """RFP (Recursive Feature Pyramid)

    This is an implementation of RFP in `DetectoRS
    <https://arxiv.org/pdf/2006.02334.pdf>`_. Different from standard FPN, the
    input of RFP should be multi level features along with origin input image
    of backbone.

    Args:
        rfp_steps (int): Number of unrolled steps of RFP.
        rfp_backbone (dict): Configuration of the backbone for RFP.
        aspp_out_channels (int): Number of output channels of ASPP module.
        aspp_dilations (tuple[int]): Dilation rates of four branches.
            Default: (1, 3, 6, 1)
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 rfp_steps,
                 rfp_backbone,
                 aspp_out_channels,
                 aspp_dilations=(1, 3, 6, 1),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.rfp_steps = rfp_steps
        # Be careful! Pretrained weights cannot be loaded when use
        # nn.ModuleList
        self.rfp_modules = ModuleList()
        for rfp_idx in range(1, rfp_steps):
            rfp_module = build_backbone(rfp_backbone)
            self.rfp_modules.append(rfp_module)
        self.rfp_aspp = ASPP(self.out_channels, aspp_out_channels,
                              aspp_dilations)
        self.softmaxattention=softmaxattention1(self.out_channels,dict(type='DCN', deform_groups=1))
        self.rfp_weight = nn.Conv2d(
            self.out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def init_weights(self):
        # Avoid using super().init_weights(), which may alter the default
        # initialization of the modules in self.rfp_modules that have missing
        # keys in the pretrained checkpoint.
        for convs in [self.lateral_convs, self.fpn_convs]:
            for m in convs.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        for rfp_idx in range(self.rfp_steps - 1):
            self.rfp_modules[rfp_idx].init_weights()
        constant_init(self.rfp_weight, 0)

    def forward(self, inputs):
        inputs = list(inputs)
        assert len(inputs) == len(self.in_channels) +1 # +1 for input image
        img = inputs.pop(0)  # imgΪͼƬ
        # FPN forward
        x = super().forward(tuple(inputs))  #    һ  ΪFPN      ʱ      һ     ػ 
        for rfp_idx in range(self.rfp_steps - 1):
            # rfp_feats = [x[0]] + list(self.rfp_aspp(x[i]) for i in range(1, len(x)))
            rfp_feats = [x[0]] + list(self.softmaxattention(x[i]) for i in range(1, len(x)))
            # rfp_feats = [x[0]] + list(x[i] for i in range(1, len(x)))
            x_idx = self.rfp_modules[rfp_idx].rfp_forward(img, rfp_feats)  #   ӷ           detectios_resnet н   ִ  
            # FPN forward
            x_idx = super().forward(x_idx) #   δ    ǰ  FPN   
            x_new = []
            for ft_idx in range(len(x_idx)):
                add_weight = torch.sigmoid(self.rfp_weight(x_idx[ft_idx]))
                x_new.append(add_weight * x_idx[ft_idx] +
                             (1 - add_weight) * x[ft_idx]) #x[ft_idx]Ϊԭ    ֵ  x_idx[ft_idx]Ϊ    backbone       ֵ            
        return x_new

