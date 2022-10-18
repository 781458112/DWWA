# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import Sequential, load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
import torch
from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import BasicBlock
from .xiaobo_resnet_s import Bottleneck as _Bottleneck
from .resnet import ResNet
from .xiaobo_resnet_s import xiaobo_ResNet_switchsimple
from .xiaobo_resnet_s import softmaxattention1 as softmaxattention_
from pytorch_wavelets import DWTForward,DTCWTInverse
import torch.nn.functional as F
def conv1x1(in_planes, out_planes, stride=1,dilation=0):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,padding=dilation,bias=False)
class softmaxattention1(softmaxattention_):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,channel,dcn):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(softmaxattention1, self).__init__(channel,dcn)
        self.relu = nn.ReLU(inplace=True)
        self.cv7=conv1x1(256,4096)
    def rfp_forward(self, x,rfp_feat,i):
        x = super().forward(x)
            # print('rfp_feat[k]',rfp_feat[k].shape,'out',out.shape)
        rfp_feat=self.cv7(rfp_feat[i])

        out = x + rfp_feat

        out = self.relu(out)

        #print('y_last_', y_last_.shape)
        return out


class Bottleneck(_Bottleneck):
    r"""Bottleneck for the ResNet backbone in `DetectoRS
    <https://arxiv.org/pdf/2006.02334.pdf>`_.

    This bottleneck allows the users to specify whether to use
    SAC (Switchable Atrous Convolution) and RFP (Recursive Feature Pyramid).

    Args:
         inplanes (int): The number of input channels.
         planes (int): The number of output channels before expansion.
         rfp_inplanes (int, optional): The number of channels from RFP.
             Default: None. If specified, an additional conv layer will be
             added for ``rfp_feat``. Otherwise, the structure is the same as
             base class.
         sac (dict, optional): Dictionary to construct SAC. Default: None.
         init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 rfp_inplanes=None,
                 sac=None,
                 init_cfg=None,
                 **kwargs):
        super(Bottleneck, self).__init__(
            inplanes, planes, init_cfg=init_cfg, **kwargs)

        assert sac is None or isinstance(sac, dict)
        self.sac = sac
        self.with_sac = sac is not None

        if self.with_sac:
            self.conv2 = build_conv_layer(
                self.sac,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                bias=False)
        if self.conv2_stride==2:
            self.conv2 = build_conv_layer(
                self.sac,
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=self.dilation,
                dilation=self.dilation,
                bias=False)
        self.rfp_inplanes = rfp_inplanes
        self.xfm = DWTForward(J=1, wave='db1', mode='zero')
        if self.rfp_inplanes:
            self.rfp_conv = build_conv_layer(
                None,
                self.rfp_inplanes,
                planes * self.expansion,
                1,
                stride=1,
                bias=True)
            if init_cfg is None:
                self.init_cfg = dict(
                    type='Constant', val=0, override=dict(name='rfp_conv'))
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')

    def rfp_forward(self, x, rfp_feat ,i):
        """The forward function that also takes the RFP features as input."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)
            if self.conv2_stride == 2:
                coeffs_yl, _ = self.xfm(out)
                out = self.conv2(coeffs_yl)
                # out = self.conv2(out)
                out = self.norm2(out)
                out = self.relu(out)
            else:
                out = self.conv2(out)
                out = self.norm2(out)
                out = self.relu(out)

                # out = self.conv2(out)
                # out = self.norm2(out)
                # out = self.relu(out)

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

        if self.rfp_inplanes:
            # i=i-1
            for k in range(i,5):
                # print('rfp_feat[k]',rfp_feat[k].shape,'out',out.shape)
                rfp_feat_1 = self._upsample_add(rfp_feat[k], out)
                rfp_feats = self.rfp_conv(rfp_feat_1)
                out = out + rfp_feats

            #rfp_feat = self.rfp_conv(rfp_feat)
            #out = out + rfp_feat

        out = self.relu(out)

        return out


class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone for RPF in detectoRS.

    The difference between this module and base class is that we pass
    ``rfp_inplanes`` to the first block.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
        rfp_inplanes (int, optional): The number of channels from RFP.
            Default: None. If specified, an additional conv layer will be
            added for ``rfp_feat``. Otherwise, the structure is the same as
            base class.
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 rfp_inplanes=None,
                 **kwargs):
        self.block = block
        assert downsample_first, f'downsample_first={downsample_first} is ' \
                                 'not supported in DetectoRS'

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                rfp_inplanes=rfp_inplanes,
                **kwargs))
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))

        super(ResLayer, self).__init__(*layers)


@BACKBONES.register_module()
class DWWA_Net(xiaobo_ResNet_switchsimple):
    """ResNet backbone for DetectoRS.

    Args:
        sac (dict, optional): Dictionary to construct SAC (Switchable Atrous
            Convolution). Default: None.
        stage_with_sac (list): Which stage to use sac. Default: (False, False,
            False, False).
        rfp_inplanes (int, optional): The number of channels from RFP.
            Default: None. If specified, an additional conv layer will be
            added for ``rfp_feat``. Otherwise, the structure is the same as
            base class.
        output_img (bool): If ``True``, the input image will be inserted into
            the starting position of output. Default: False.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 sac=None,
                 stage_with_sac=(False, False, False, False),
                 rfp_inplanes=None,
                 output_img=False,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        self.pretrained = pretrained
        if init_cfg is not None:
            assert isinstance(init_cfg, dict), \
                f'init_cfg must be a dict, but got {type(init_cfg)}'
            if 'type' in init_cfg:
                assert init_cfg.get('type') == 'Pretrained', \
                    'Only can initialize module by loading a pretrained model'
            else:
                raise KeyError('`init_cfg` must contain the key "type"')
            self.pretrained = init_cfg.get('checkpoint')
        self.sac = sac
        self.stage_with_sac = stage_with_sac
        self.rfp_inplanes = rfp_inplanes
        self.output_img = output_img
        super(DWWA_Net, self).__init__(**kwargs)

        self.inplanes = self.stem_channels
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            sac = self.sac if self.stage_with_sac[i] else None
            if self.plugins is not None:
                stage_plugins = self.make_stage_plugins(self.plugins, i)
            else:
                stage_plugins = None
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                sac=sac,
                rfp_inplanes=rfp_inplanes if i > 0 else None,
                plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.add_module('layer5', self.layer5)  ###
        self.res_layers.append('layer5')  ####
        self._freeze_stages()
        # self.norm1=nn.BatchNorm2d(64)
        self.softmaxattention1=softmaxattention1(channel=2048,dcn=dict(type='DCN', deform_groups=1))
        # self.cv1=conv1x1(2048,4096,stride=2)
    # In order to be properly initialized by RFP
    def init_weights(self):
        # Calling this method will cause parameter initialization exception
        # super(DetectoRS_ResNet, self).init_weights()

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer`` for DetectoRS."""
        return ResLayer(**kwargs)

    def forward(self, x):
        """Forward function."""
        outs = list(super(DWWA_Net, self).forward(x))
        if self.output_img:
            outs.insert(0, x)
        return tuple(outs)

    def rfp_forward(self, x, rfp_feats):
        """Forward function for RFP."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            coeffs, coeffs_ = self.xfm(x)
            coeffs_yh, coeffs_yh1, coeffs_yh2 = coeffs_[0][:, :, 0, :, :], coeffs_[0][:, :, 1, :, :], coeffs_[0][:, :,
                                                                                                      2, :, :]
            kkk = torch.cat([coeffs, coeffs_yh, coeffs_yh1,
                             coeffs_yh2],
                            dim=1)  ##xiaobo
            x = self.conv1(x)
            x = torch.cat([x, kkk], dim=1)
            x = self.conv2(x)
            x = self.norm1(x)
            x = self.relu(x)
            # x = torch.cat([x, kkk], dim=1)
            # x,_ = self.xfm(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            # i=i+1
            res_layer = getattr(self, layer_name)
            rfp_feat = rfp_feats if i > 0 else None
            if layer_name == 'layer5':
                x =self.softmaxattention1.rfp_forward(x, rfp_feat,i)
                outs.append(x)
            else:
                for layer in res_layer:
                    x = layer.rfp_forward(x, rfp_feat,i)
                if i in self.out_indices:
                    outs.append(x)
        return tuple(outs)
