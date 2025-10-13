from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from mmdet3d_plugin.datasets.utils import box3d_to_corners
from mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk


@HEADS.register_module()
class FocalADMotionPlanningHead(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
        use_tp=None,
        motion_feature_refine =None
    ):
        super(FocalADMotionPlanningHead, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        if plan_anchor is not None:
            plan_anchor = np.load(plan_anchor)
            self.plan_anchor = nn.Parameter(
                torch.tensor(plan_anchor, dtype=torch.float32),
                requires_grad=False,
            )
            self.plan_anchor_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 1),
                Linear(embed_dims, embed_dims),
            )
        else:
            self.plan_mode_embedding = nn.Embedding(self.ego_fut_mode, embed_dims)

        self.num_det = num_det
        self.num_map = num_map
        self.use_tp = use_tp
        if self.use_tp is not None:
            self.tp_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 2)
            )
        # =========== build new block ===========
        self.motion_feature_refine = build(motion_feature_refine, PLUGIN_LAYERS)

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self, 
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):   
        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )

        map_instance_feature = map_output["instance_feature"]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )

        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)

        # =========== motion init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))

        # =========== plan init ===========
        if hasattr(self, "plan_anchor"):
            plan_anchor = torch.tile(
                self.plan_anchor[None], (bs, 1, 1, 1, 1)
            )
            plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
            plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)
        else:
            plan_mode_query = torch.tile(
                self.plan_mode_embedding.weight[None, None], (bs, 1, 1, 1)
            )

        # =========== FocalAD Block ===========

        ego_node_raw = ego_anchor[...,[X, Y, W, L, SIN_YAW,COS_YAW, VX, VY]]
        n_node_raw =  det_output["prediction"][-1][...,[X, Y, W, L, SIN_YAW, COS_YAW, VX, VY,]]
        ego_pos = ego_anchor[...,[X, Y]]
        n_pos = det_output["prediction"][-1][...,[X, Y]]
        ego_vel = ego_anchor[...,[VX, VY]]
        n_vel = det_output["prediction"][-1][...,[VX, VY]]
        ego_dir = ego_anchor[..., [COS_YAW, SIN_YAW]]
        n_dir = det_output["prediction"][-1][..., [COS_YAW, SIN_YAW]]
        cos_delta = torch.sum(ego_dir.expand(-1, n_dir.size(1),-1) * n_dir, dim=-1)
        pos_diff = ego_pos - n_pos
        vel_diff = ego_vel - n_vel
        # Euclidean Distance
        pos_dist = torch.norm(pos_diff, dim=-1)
        vel_dist = torch.norm(vel_diff, dim=-1)

        node_raw = torch.cat([
            ego_node_raw,
            n_node_raw,
        ], dim = 1)
        edge_raw = torch.cat([
            pos_dist.unsqueeze(-1),
            cos_delta.unsqueeze(-1),
            vel_dist.unsqueeze(-1)
        ], dim=-1)
        # E = len(n_node_raw[1])
        ego_id = 0

        h_ego_prime, selected_feat, topk_idx, alpha = self.motion_feature_refine(node_raw, edge_raw, ego_id)

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                if self.use_tp:
                    tp_embedding = self.tp_encoder(metas["tp_near"].float())
                    instance_feature[:, num_anchor:] = instance_feature[:, num_anchor:] + tp_embedding[:, None]

                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                # plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2)
                # =========== Query Refinement===========
                B, N, T, D = motion_query.shape

                # 构造 scatter 索引：只在 dim=1 上 scatter
                index = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T ,D)  # (B, K, T,D)

                # 新建更新量
                update = torch.zeros_like(motion_query)  # (B, N, T ,D)

                # 执行 scatter_add_
                update.scatter_add_(dim=1, index=index, src=selected_feat)

                # 加回最后一帧
                motion_query = motion_query + update
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2) + h_ego_prime.unsqueeze(1).unsqueeze(2)
                
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,
                ) = self.layers[i](
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                    metas,
                )
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
        
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        
        planning_output = {
            "classification": planning_classification, #1216
            "prediction": planning_prediction,#1216 62
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output
    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache,
        topk_idx,
        alpha,
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache, topk_idx, alpha )
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache, topk_idx, alpha ):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
                topk_idx,
                alpha,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]#[1,2,1,6]
        reg_preds = model_outs["prediction"]     #[1,2,1,6,6,2]
        status_preds = model_outs["status"]      #[1,2,1,10]
        output = {}

        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            
            (
                cls,#torch.Size([1, 1, 6])
                cls_target, #torch.Size([1, 1])
                cls_weight, #torch.Size([1, 1])mask
                reg_pred, #torch.Size([1, 1, 6, 2])
                reg_target, #torch.Size([1, 1, 6, 2])
                reg_weight, #torch.Size([1, 1, 6])mask
            ) = self.planning_sampler.sample(
                cls,#torch.Size([1, 1, 6])
                reg,#torch.Size([1, 1, 6, 6, 2])
                data['gt_ego_fut_trajs'],#torch.Size([1, 6, 2])
                data['gt_ego_fut_masks'],#torch.Size([1, 6])
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            # print(reg_target.sum(dim=1))
            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output, 
            data,
        )

        return motion_result, planning_result