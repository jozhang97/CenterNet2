import math
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, get_norm
from ..layers.deform_conv import DFConv2d

__all__ = ["CenterNetHead"]

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class CenterNetHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        self.with_agn_hm = cfg.MODEL.CENTERNET.WITH_AGN_HM
        self.only_proposal = cfg.MODEL.CENTERNET.ONLY_PROPOSAL
        self.out_kernel = 3
        norm = cfg.MODEL.CENTERNET.NORM

        head_configs = {"cls": (cfg.MODEL.CENTERNET.NUM_CLS_CONVS \
                                if not self.only_proposal else 0,
                                cfg.MODEL.CENTERNET.USE_DEFORMABLE),
                        "bbox": (cfg.MODEL.CENTERNET.NUM_BOX_CONVS,
                                 cfg.MODEL.CENTERNET.USE_DEFORMABLE),
                        "share": (cfg.MODEL.CENTERNET.NUM_SHARE_CONVS,
                                  cfg.MODEL.CENTERNET.USE_DEFORMABLE)}

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, \
            "Each level must have the same channel!"
        in_channels = in_channels[0]
        channels = {
            'cls': in_channels,
            'bbox': in_channels,
            'share': in_channels,
        }
        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            channel = channels[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d
                tower.append(conv_func(
                        in_channels if i == 0 else channel,
                        channel, 
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                ))
                if norm == 'GN' and channel % 32 != 0:
                    tower.append(nn.GroupNorm(25, channel))
                elif norm != '':
                    tower.append(get_norm(norm, channel))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in input_shape])

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower,
            self.bbox_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        
        torch.nn.init.constant_(self.bbox_pred.bias, 8.)
        prior_prob = cfg.MODEL.CENTERNET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        if self.with_agn_hm:
            self.agn_hm = nn.Conv2d(
                in_channels, 1, kernel_size=self.out_kernel,
                stride=1, padding=self.out_kernel // 2
            )
            torch.nn.init.constant_(self.agn_hm.bias, bias_value)
            torch.nn.init.normal_(self.agn_hm.weight, std=0.01)

        self.use_reid = cfg.MODEL.CENTERNET.USE_REID
        self.reid_dim = cfg.MODEL.CENTERNET.REID_DIM
        if self.use_reid:
            self.reid_hm = nn.Conv2d(
                in_channels, self.reid_dim, kernel_size=self.out_kernel,
                stride=1, padding=self.out_kernel // 2
            )

        if not self.only_proposal:
            cls_kernel_size = self.out_kernel
            self.cls_logits = nn.Conv2d(
                in_channels, self.num_classes,
                kernel_size=cls_kernel_size, 
                stride=1,
                padding=cls_kernel_size // 2,
            )

            torch.nn.init.constant_(self.cls_logits.bias, bias_value)
            torch.nn.init.normal_(self.cls_logits.weight, std=0.01)


    def forward(self, x):
        clss = []
        bbox_reg = []
        agn_hms = []
        reid_hms = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if not self.only_proposal:
                clss.append(self.cls_logits(cls_tower))
            else:
                clss.append(None)

            if self.with_agn_hm:
                agn_hms.append(self.agn_hm(bbox_tower))
            else:
                agn_hms.append(None)
            reg = self.bbox_pred(bbox_tower)
            reg = self.scales[l](reg)
            bbox_reg.append(F.relu(reg))
            if self.use_reid:
                reid_hms.append(self.reid_hm(bbox_tower))
            else:
                reid_hms.append(None)
        return clss, bbox_reg, agn_hms, reid_hms
