# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import pywt
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import torch
import torch.nn.functional as F
from ..builder import BACKBONES
from ..utils import ResLayer
def conv1x1(in_planes, out_planes, stride=1,dilation=0):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=dilation,bias=False)
def conv1x1_down(in_planes, out_planes, stride=2,dilation=0):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=dilation,bias=False)
def conv3x3(in_planes, out_planes, stride=1,dilation=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation,bias=False)
def conv5x5(in_planes, out_planes, stride=1,dilation=2):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,padding=dilation,bias=False)
def conv3x3_down(in_planes, out_planes, stride=2,dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False)
class SELayerada(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayerada, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.dropput=F.dropout(p=0.5),
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Dropout(p=0.5),
            # F.dropout(p=0.5),
            # nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class SELayermax(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayermax, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Dropout(p=0.5),
            # nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.max_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class softmaxattention1(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,channel,dcn, k=1, s=1, p=None):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(softmaxattention1, self).__init__()
        # hidden channels
        #print('c1',c1/8)
        # self.cv1 = Conv(2048,256, 3, 1,1)#640*640*9
        # self.cv2 = Conv(2048,256, 5, 1,2)#640*640*6
        # self.cv1= conv3x3(1024,128, 3, 1,1)
        # self.cv2 = conv3x3(1024, 128, 3, 1, 1)
        self.se1=SELayerada(channel,reduction=16)
        self.se2 = SELayermax(channel, reduction=16)
        # self.cv1 = conv1x1(2048, 2048)  # 640*640*9
        # self.cv2 = conv3x3(2048, 2048)  # 640*640*6
        # self.cv3 = conv1x1(2048,2048,stride=1)
        # self.cv4 = conv1x1_down(4096, 4096)
        # self.cv5 =conv1x1(12288,4096)
        self.dcn = dcn
        # if self.dcn:
        #     fallback_on_stride = dcn.pop('fallback_on_stride', False)
        self.cv1 = build_conv_layer(dcn,2048,2048,kernel_size=1,stride=1,padding=0,dilation=1,bias=False)
        self.cv2 = build_conv_layer(dcn, 2048, 2048, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.cv3 = build_conv_layer(dcn, 2048, 2048, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cv4 = build_conv_layer(dcn, 4096, 4096, kernel_size=1, stride=2, padding=0, dilation=1, bias=False)
        self.cv5 = build_conv_layer(dcn, 12288, 4096, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        #self.cv3 = Conv(6, 3,3, 1,1)  #640*640*3# act=FReLU(c2)
        #Conv(3, 3, 1, 1, 0)
        #self.m 6= nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
    def forward(self, x):
        coeffs = pywt.dwt2(x.cpu().detach().numpy(), 'db1')
        kkk = torch.cat([torch.tensor(coeffs[0]), torch.tensor(coeffs[1][0]), torch.tensor(coeffs[1][1]),
                         torch.tensor(coeffs[1][2])], dim=1).cuda()

        #print('x1',x.shape)
        y1=self.cv1(x)  ##[B,C,H,W]  C
        #print('y1', y1.shape)
        y2=self.cv2(x)  ##  B
        y1se=self.se1(y1)
        y2se=self.se2(y2)
        y3=self.cv3(x)
        #print('y2', y2.shape)##[B,C,H,W]
        # x_= x.view([x.size()[0], x.size()[1], -1])#[B,C,H*w]
        #print('x_', x_.shape)  ##[B,C,H,W]

        y2_=y2se.view([y2.size()[0],y2.size()[1],-1]) #[B,C,H*w] 0 1 2
        #print('y2_', y2_.shape)  ##[B,C,H*w] 0 1 2
        y1_ = y1se.view([y1.size()[0], y1.size()[1], -1])  # [B,C,H*w] 0 1 2
        #print('y1_ ', y1_ .shape)  ##[B,C,H*w] 0 1 2

        y2_T = y2_.permute([0,2,1])  # [B,H*w,C]  0 2 1  D   ###K
        #print('y2_T ', y2_T.shape)# [B,H*w,C]
        #c=y1_*y2_T  #[B,C,H*w]  * [B,H*w,C]
        c=torch.matmul(y2_T,y1_ ) #[B,H*w,H*w]  ###QK
        #print('c ', c.shape)  # [B,H*w,H*w]
        C_weight=F.softmax(c,dim=-1)   ##[B,H*w,H*w]  E
        #print('C_weight', C_weight.shape)  # [B,H*w,H*w]
        #y=C*x
        y3_=y3.view([x.size()[0], x.size()[1], -1])
        y_last =torch.matmul(y3_,C_weight)
        y_last = torch.nn.functional.normalize(y_last)
        # y_last = torch.matmul(x_,C_weight) #[B,C,H*w] *[B,H*w,H*w]  F-  ###yuanshi
        y_last_ = y_last.view([x.size()[0], x.size()[1], x.size()[2], x.size()[3]])  # [B,C,H*w] 0 1 2
        # y_last_ = torch.nn.functional.normalize(y_last_)
        # zhao=torch.add(y_last_, x)
        zhao=torch.cat((y_last_,x),dim=1)
        zhao=self.cv4(zhao)
        kkk_=torch.cat((zhao,kkk),dim=1) #³¢ÊÔÏÂadd
        zhao_new = self.cv5(kkk_)
        # y_last=y_last.permute([0,1,3,2])
        # y_last_.permute(0,1,3,2).contiguous()
        # zhao = torch.matmul(y_last[::-1], x)  ##y_last [B,C,H,w]   x[B,C,H,W]
        # print('y1', y1.shape)
        # y2=self.cv2(y1)

        #print('y_last_', y_last_.shape)
        return zhao_new
class softmaxattention(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,dcn=None, p=None):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(softmaxattention, self).__init__()
        # hidden channels
        #print('c1',c1/8)
        self.cv1 = conv1x1(2048,2048)#640*640*9
        self.cv2 = conv1x1(2048,2048)#640*640*6
        self.cv3 = conv1x1(2048, 2048)  # 640*640*6
        self.cv4 = conv1x1(4096, 4096)  # 640*640*6
        self.cvdown=conv3x3(2048,2048)
        # self.cv5=conv1x1(2048,2048)
        # self.dcn=dcn
        # if self.dcn:
        #     fallback_on_stride = dcn.pop('fallback_on_stride', False)
        # self.dcn=build_conv_layer(
        #         dcn,
        #         2048,
        #         2048,
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #         dilation=1,
        #         bias=False)
        # self.se1=SELayerada(channel,reduction=16)
        # self.se2 = SELayermax(channel, reduction=16)
    def forward(self, x):
        #print('x1',x.shape)
        y1=self.cv1(x)  ##[B,C,H,W]  C
        #print('y1', y1.shape)
        y2=self.cv2(x)  ##  B
        y3=self.cv3(x)
        # y5=self.cv5(x)
        #print('y2', y2.shape)##[B,C,H,W]
        # x_= x.view([x.size()[0], x.size()[1], -1])#[B,C,H*w]
        #print('x_', x_.shape)  ##[B,C,H,W]
        # coeffs = pywt.dwt2(x.cpu().detach().numpy(), 'db1')
        # kkk = torch.cat([torch.tensor(coeffs[0]), torch.tensor(coeffs[1][0]), torch.tensor(coeffs[1][1]),
        #                  torch.tensor(coeffs[1][2])], dim=1).cuda()
        y2_=y2.view([y2.size()[0],y2.size()[1],-1]) #[B,C,H*w] 0 1 2
        #print('y2_', y2_.shape)  ##[B,C,H*w] 0 1 2
        y1_ = y1.view([y1.size()[0], y1.size()[1], -1])  # [B,C,H*w] 0 1 2
        #print('y1_ ', y1_ .shape)  ##[B,C,H*w] 0 1 2

        y2_T = y2_.permute([0,2,1])  # [B,H*w,C]  0 2 1  D   ###K
        #print('y2_T ', y2_T.shape)# [B,H*w,C]
        #c=y1_*y2_T  #[B,C,H*w]  * [B,H*w,C]
        c=torch.matmul(y2_T,y1_ ) #[B,H*w,H*w]  ###QK
        y3_ = y3.view([y3.size()[0], y3.size()[1], -1])
        y3_T =y3_.permute([0,2,1])
        # y4=self.cv4(coeffs)
        d=torch.matmul(c,y3_T)
        d_weight = F.softmax(d, dim=-1)
        # y3_=y3.view([x.size()[0], x.size()[1], -1])
        # y_last =torch.matmul(y3_,d_weight)
        d_=d.permute([0,2,1]).view([x.size()[0], x.size()[1], x.size()[2], x.size()[3]])
        # y_last_=self.cvdown(d_)
        # y_finally=y_last_.view([y_last_.size()[0], y_last_.size()[1], -1])
        # y_finally_T=y_finally.permute([0,2,1])
        # k_finally=kkk.view([kkk.size()[0], kkk.size()[1], -1]).cuda()
        # f=torch.matmul(y_finally_T, k_finally)
        # f = torch.nn.functional.normalize(f)
        # k_last = torch.cat((y_last_,y5), dim=1)
        # k_last=torch.cat((y_last_,kkk,y5),dim=1)
        # y_last_y = f.view([x.size()[0], x.size()[1], x.size()[2], x.size()[3]])  # [B,C,H*w] 0 1 2
        zhao = torch.cat((d_, x), dim=1)
        zhao = self.cv4(zhao)
        #zhao = self.cv4(d)
        # zhao=torch.add((zhao,x),dim=1)
        return zhao

class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class xiaobo_ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super(xiaobo_ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn

        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)
        self.layer5 = softmaxattention1(channel=2048,dcn=dcn)
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.add_module('layer5', self.layer5)###
        self.res_layers.append('layer5')####
        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2=conv1x1(76,64)
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            coeffs = pywt.dwt2(x.cpu().detach().numpy(), 'db1')
            kkk = torch.cat([torch.tensor(coeffs[0]), torch.tensor(coeffs[1][0]), torch.tensor(coeffs[1][1]),
                             torch.tensor(coeffs[1][2])],
                            dim=1).cuda()  ##xiaobo
            x = self.conv1(x)
            x = torch.cat([x, kkk], dim=1)
            x = self.conv2(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
            if layer_name=='layer5':
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(xiaobo_ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module()
class xiaobo_ResNetV1d(xiaobo_ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(xiaobo_ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
