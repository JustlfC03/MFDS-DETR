# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy

# 已删除DCNv2文件夹，使用torchvision.ops替代
# from dcn_v2 import DCN as dcn_v2
from torchvision.ops import DeformConv2d as dcn_v2

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, with_fpn=False, method_fpn=""):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            with_fpn: 是否使用了FPN来进行特征融合
            method_fpn: 用了什么方式来融合(fpn, bifpn, fapn, pafpn)
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.with_fpn = with_fpn
        self.method_fpn = method_fpn

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)

        #多尺度的特征图输入
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            if not self.with_fpn:
                for _ in range(num_feature_levels - num_backbone_outs):
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                    in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        if self.with_fpn:
            in_channel = backbone.num_channels[-1]
            # 获得最高层的feature map (这里都是默认为向上提取一层)
            self.top_feature_proj = nn.ModuleList()
            for i in range(num_feature_levels - num_backbone_outs):
                self.top_feature_proj.append(nn.Sequential(
                    nn.Conv2d(in_channel, in_channel//2, kernel_size=1),
                    nn.Conv2d(in_channel//2, in_channel//2, kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(in_channel//2, in_channel*2, kernel_size=1),
                    nn.GroupNorm(32, in_channel*2),
                ))
            # proj list
            self.fpn_proj_list = nn.ModuleList()
            for _ in range(num_feature_levels):
                self.fpn_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))

            if self.method_fpn == "fpn":
                self.bottomup_conv1 = nn.ModuleList()
                for _ in range(num_feature_levels - 2, -1, -1):
                    in_channels = backbone.num_channels[_]
                    self.bottomup_conv1.append(nn.Sequential(
                        nn.Conv2d(in_channels, in_channel, kernel_size=1),
                        nn.GroupNorm(32, in_channel)
                    ))


            if self.method_fpn == "pafpn":
                self.bottomup_conv = nn.ModuleList()
                self.upbottom_conv = nn.ModuleList()
                for i in range(num_feature_levels - 1):
                    in_channels = backbone.num_channels[num_feature_levels - i - 2]
                    self.upbottom_conv.append(nn.Sequential(
                        nn.Conv2d(in_channels, in_channel, kernel_size=1),
                        nn.GroupNorm(32, in_channel),
                    ))
                    self.bottomup_conv.append(nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, in_channel),
                    ))

                for c1, c2 in zip(self.bottomup_conv, self.upbottom_conv):
                    nn.init.xavier_uniform_(c1[0].weight, gain=1)
                    nn.init.constant_(c1[0].bias, 0)
                    nn.init.xavier_uniform_(c2[0].weight, gain=1)
                    nn.init.constant_(c2[0].bias, 0)

            if self.method_fpn == "bifpn":
                self.upbottom_conv = nn.ModuleList()
                self.bottomup_conv1 = nn.ModuleList()
                self.bottomup_conv2 = nn.ModuleList()
                for i in range(num_feature_levels - 1):
                    in_channels = backbone.num_channels[num_feature_levels - i - 2]
                    if i + 1 == num_feature_levels - 1:
                        in_channels_cross = in_channel*2
                    else:
                        in_channels_cross = backbone.num_channels[i+1]
                    self.upbottom_conv.append(nn.Sequential(
                        nn.Conv2d(in_channels, in_channel, kernel_size=1),
                        nn.GroupNorm(32, in_channel),
                    ))
                    self.bottomup_conv1.append(nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, in_channel),
                        nn.ReLU(inplace=True)
                    ))
                    self.bottomup_conv2.append(nn.Sequential(
                        nn.Conv2d(in_channels_cross, in_channel, kernel_size=3, stride=1, padding=1),
                        nn.GroupNorm(32, in_channel),
                        nn.ReLU(inplace=True)
                    ))

                for c1, c2, c3 in zip(self.upbottom_conv, self.bottomup_conv1, self.bottomup_conv2):
                    nn.init.xavier_uniform_(c1[0].weight, gain=1)
                    nn.init.constant_(c1[0].bias, 0)
                    nn.init.xavier_uniform_(c2[0].weight, gain=1)
                    nn.init.constant_(c2[0].bias, 0)
                    nn.init.xavier_uniform_(c3[0].weight, gain=1)
                    nn.init.constant_(c3[0].bias, 0)

            if self.method_fpn == "fapn":
                self.align_modules = nn.ModuleList()
                self.bottomup_conv = nn.ModuleList()
                for i in range(num_feature_levels - 1):
                    in_channels = backbone.num_channels[num_feature_levels - i - 2]
                    align_module = FeatureAlign_V2(in_channels, hidden_dim)
                    self.align_modules.append(align_module)
                for i in range(num_feature_levels):
                    self.bottomup_conv.append(nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                        nn.ReLU()
                    ))

            if self.method_fpn == "wbcfpn":
                self.upbottom_conv = nn.ModuleList()
                self.weight_conv = nn.ModuleList()
                self.lateral_conv = nn.ModuleList()
                # self.bottomup_lateral_conv = nn.ModuleList()
                # self.up_conv = nn.ModuleList()
                # self.cbam_attention = nn.ModuleList()
                self.up_sample_conv = nn.ModuleList()
                for i in range(num_feature_levels):
                    if i == 0:
                        in_channels = in_channel*2
                    else:
                        in_channels = backbone.num_channels[num_feature_levels - i - 1]
                    self.upbottom_conv.append(nn.Sequential(
                        ChannelAttention(in_channels),
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                    ))
                    if i != 0:
                        self.weight_conv.append(nn.Sequential(
                            ChannelAttention(hidden_dim, ratio=4, flag=False)
                        ))
                        self.up_sample_conv.append(nn.Sequential(
                            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2,
                                               padding=1, output_padding=1),
                        ))
                    self.lateral_conv.append(nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                        nn.ReLU(),
                    ))
                    # self.bottomup_lateral_conv.append(nn.Sequential(
                    #     SpatialAttention(),
                    # ))

                    # if i != 0:
                    #     self.up_conv.append(nn.Sequential(
                    #         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                    #     ))
                        # self.up_sample_conv.append(nn.Sequential(
                        #     nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2,
                        #                        padding=1, output_padding=1),
                        # ))

                    # cbam_attention = nn.Sequential(
                    #     ChannelAttention(hidden_dim),nn
                    #     SpatialAttention(),
                    #     nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    #     nn.GroupNorm(32, hidden_dim),
                    # )
                    # self.cbam_attention.append(cbam_attention)
                    # self.add_module("cbam_attention_{}".format(i), self.cbam_attention)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def _upsample_add(self, x, y):
        """
        上采样并且将两个feature map进行相加
        Parameters:
            x: 上层的feature map
            y: 融合的feature map
        """
        _, _, h, w = y.size()
        return F.upsample(x, size=(h, w), mode='bilinear') + y

    def fpn(self, srcs):
        """
        最普通的FPN方式，通过element wise方式将不同层的feature map进行相加
        Parameters:
            srcs:不同层的feature map
        """
        # reverse srcs
        feature_maps = srcs[::-1]
        results = [feature_maps[0]]
        prev_feature = feature_maps[0]
        for feature, conv in zip(feature_maps[1:], self.bottomup_conv1):
            prev_feature = self._upsample_add(prev_feature, conv(feature))
            results.insert(0, prev_feature)
        return results

    def pafpn(self, srcs):
        """
        这是PANet中特征融合的方式：
        paper: https://arxiv.org/abs/1803.01534
        code: https://github.com/ShuLiu1993/PANet
        """
        # reverse srcs
        feature_maps = srcs[::-1]
        up_bottom_features = []
        bottom_up_features = []
        up_bottom_features.append(feature_maps[0])
        # 从上到下的特征融合
        for feature, conv in zip(feature_maps[1:], self.upbottom_conv):
            prev_feature = self._upsample_add(up_bottom_features[0], conv(feature))
            up_bottom_features.insert(0, prev_feature)

        bottom_up_features.append(up_bottom_features[0])
        for i in range(1, len(up_bottom_features)):
            prev_feature = self.bottomup_conv[i - 1](bottom_up_features[0])
            prev_feature = prev_feature + up_bottom_features[i]
            bottom_up_features.insert(0, prev_feature)

        return bottom_up_features[::-1]

    def bifpn(self, srcs):
        """
        这是EfficientDet的特征融合方式:
        paper: https://arxiv.org/abs/1911.09070
        code: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
        """
        # reverse srcs
        feature_maps = srcs[::-1]
        up_bottom_features = []
        bottom_up_features = []
        up_bottom_features.append(feature_maps[0])
        for feature, conv in zip(feature_maps[1:], self.upbottom_conv):
            prev_feature = self._upsample_add(up_bottom_features[0], conv(feature))
            up_bottom_features.insert(0, prev_feature)


        bottom_up_features.append(up_bottom_features[0])
        for i in range(1, len(up_bottom_features)):
            prev_feature = self.bottomup_conv1[i-1](bottom_up_features[0])
            prev_feature = prev_feature + up_bottom_features[i] + \
                           self.bottomup_conv2[i-1](srcs[i])
            bottom_up_features.insert(0, prev_feature)

        return bottom_up_features[::-1]

    def fapn(self, srcs):
        """
        这是FaPN的特征融合:
        paper:
        code:https://github.com/ShihuaHuang95/FaPN-full
        """
        #reverse feature map
        feature_maps = srcs[::-1]
        results = [feature_maps[0]]
        for feature, align_module in zip(feature_maps[1:], self.align_modules):
            prev_feature = align_module(feature, results[0])
            results.insert(0, prev_feature)

        for i in range(self.num_feature_levels):
            results[i] = self.bottomup_conv[i](results[i])

        return results

    def wbcfpn(self, srcs):
        """
        这是我们自己设计的FPN
        """
        feature_maps = srcs[::-1]
        up_sample_features = []
        up_bottom_features = []
        #bottom_up_features = []
        up_bottom_features.append(self.upbottom_conv[0](feature_maps[0]))
        up_sample_features.append(up_bottom_features[0])
        for up_sample in self.up_sample_conv:
            up_sample_features.insert(0, up_sample(up_sample_features[0]))

        up_sample_features = up_sample_features[::-1]

        for i, (feature, conv, weight_conv) in enumerate(zip(feature_maps[1:], self.upbottom_conv[1:], self.weight_conv)):
            down_feature = conv(feature)
            _, _, h, w = feature.shape
            high_feature = up_sample_features[i+1]
            if high_feature.shape[-1] != w or high_feature.shape[-2] != h:
                high_feature = F.upsample(high_feature, size=(h, w), mode='bilinear')

            select_down_feature = weight_conv(high_feature)*down_feature
            fusion_feature = select_down_feature + high_feature
            up_bottom_features.append(fusion_feature)

        results = up_bottom_features[::-1]
        for i in range(len(results)):
            results[i] = self.lateral_conv[i](results[i])
        #
        # for feature, conv, up_sample in zip(feature_maps[1:], self.upbottom_conv[1:], self.up_sample_conv):
        #     up_feature = up_sample(up_bottom_features[0])
        #     prev_feature = self._upsample_add(up_feature, conv(feature))
        #     up_bottom_features.insert(0, prev_feature)
        #
        # bottom_up_features.append(self.bottomup_lateral_conv[0](up_bottom_features[0]))
        # for i in range(1, len(up_bottom_features)):
        #     prev_feature = self.up_conv[i-1](bottom_up_features[0])
        #     prev_feature = prev_feature + self.bottomup_lateral_conv[i](up_bottom_features[i])
        #     bottom_up_features.insert(0, prev_feature)
        #
        # results = bottom_up_features[::-1]
        # for i in range(len(results)):
        #     results[i] = self.cbam_attention[i](results[i])

        return results

    def get_fpn(self, method_fpn, srcs):
        """
        Parameters:
            method_fpn: fpn的方式
            srcs: 输入的特征图
        """
        fpn_map = {
            'fpn': self.fpn,
            'bifpn': self.bifpn,
            'pafpn': self.pafpn,
            'fapn': self.fapn,
            'wbcfpn': self.wbcfpn
        }
        assert method_fpn in fpn_map, f'do you really want to using the {method_fpn} ?'
        return fpn_map[method_fpn](srcs)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        #features: 特征图
        #pos: position embedding
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            #改变每个feature map的维度
            srcs.append(src)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if self.with_fpn:
                    if l == _len_srcs:
                        src = self.top_feature_proj[_len_srcs - l](features[-1].tensors)
                    else:
                        src = self.top_feature_proj[_len_srcs - l](srcs[-1])
                    srcs.append(src)
                else:
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)

        # 使用FPN来进行融合
        if self.with_fpn:
            srcs = self.get_fpn(self.method_fpn, srcs)
            if self.method_fpn != "fapn" and self.method_fpn != "wbcfpn":
                for i in range(len(srcs)):
                    srcs[i] = self.fpn_proj_list[i](srcs[i])

            for i in range(len(srcs) - 1, self.num_feature_levels):
                src = srcs[i]
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                masks.append(mask)
                pos.append(pos_l)
        else:
            for i in range(len(srcs) - 1):
                srcs[i] = self.input_proj[i](srcs[i])
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        #输出的格式
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        #类别数，不包括背景
        self.num_classes = num_classes
        #匈牙利算法：用于预测和GT进行匹配的算法
        self.matcher = matcher
        #对lloss对应的权重
        self.weight_dict = weight_dict
        #指定需要计算哪些loss
        self.losses = losses
        #focal loss的参数
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        # (b, num_queries, num_classes+1)
        src_logits = outputs['pred_logits']

        # idx是一个tuple，代表所有匹配的预测结果的batch index和query index
        idx = self._get_src_permutation_idx(indices)
        # 匹配的GT，(num_matched_targets1 + num_matched_targets2 + ... ,)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # (b, num_queries)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 匹配预测索引对应值设置为匹配的GT：对于每个batch的queries加入类别，没有匹配到就是num_classes
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # 返回匹配的预测结果的批次索引和queries索引
        # permute predictions following indices
        # (num_matched_queries1 + num_matched_queries2+..., )
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        """
        ouputs: Deformable DETR模型输出，是一个dict，形式如下：
                {'pred_logits': (b, num_queries, num_classes), 'pred_boxes': (b, num_queries, 4),
                'aux_outputs': [{'pred_logits':, 'pred_boxed':}]}
        """
        #过滤掉中间层的输出，只保留最后一层的预测结果
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # 将预测结果和GT进行匹配
        # indices: 是一个包含多个元组的list, 长度和batch size相等，每个元组为(index_i, index_j)
        #   index_i: 匹配的预测索引
        #   index_j: GT索引
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # 计算这个batch的图像中目标物体的数量
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            #计算特定类型的loss
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # 如果模型输出包含了中间层的输出，那么一并计算对应的loss
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 6
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        with_fpn=args.with_fpn,
        method_fpn=args.method_fpn
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    #将预测结果和GT进行匹配的算法(匈牙利算法)
    matcher = build_matcher(args)
    #各类型loss的权重
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    #weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            #为中间层输出的loss也加上对应权重
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    #指定计算哪些类型的loss
    #其中cardinality是计算预测为前景的数量与GT数量的L1误差
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 4, flag=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x if self.flag else self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, flag=True):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.flag = flag
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)*x if self.flag else self.sigmoid(out)

class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1)
        self.group_norm1 = nn.GroupNorm(32, in_chan)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1)
        self.group_norm2 = nn.GroupNorm(32, out_chan)
        nn.init.xavier_uniform_(self.conv_atten.weight)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        atten = self.sigmoid(self.group_norm1(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.group_norm2(self.conv(x))
        return feat

class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc, out_nc):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc)
        self.offset = nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0)
        self.group_norm1 = nn.GroupNorm(32, out_nc)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.offset.weight)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.group_norm1(self.offset(torch.cat([feat_arm, feat_up * 2], dim=1)))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))  # [feat, offset]
        return feat_align + feat_arm
