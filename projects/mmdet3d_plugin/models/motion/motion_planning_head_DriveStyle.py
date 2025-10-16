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
import torch.nn.functional as F

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk


from diffusers.schedulers import DDIMScheduler

try:
    from projects.mmdet3d_plugin.ops import deformable_aggregation_function as DAF
except:
    DAF = None
from projects.mmdet3d_plugin.models.motion.modules.conditional_unet1d import ConditionalUnet1D, SinusoidalPosEmb
import torch.nn.functional as F




class TrajSparsePoint3DKeyPointsGenerator(BaseModule): 
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        num_learnable_pts: int = 0,
        fix_height: Tuple = (0,),
        ground_height: int = 0,
    ):
        super(TrajSparsePoint3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts
        # if self.num_learnable_pts > 0:
        #     self.learnable_fc = Linear(self.embed_dims, self.num_pts * 2)

        self.fix_height = np.array(fix_height)
        self.ground_height = ground_height

    # def init_weight(self):
    #     if self.num_learnable_pts > 0:
    #         xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        # import ipdb; ipdb.set_trace()
        # assert self.num_learnable_pts > 0, 'No learnable pts'
        bs, num_anchor, _ = anchor.shape
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)
        offset = torch.zeros([bs, num_anchor, self.num_sample, len(self.fix_height), 1, 2],device=anchor.device, dtype=anchor.dtype)
      
        key_points = offset + key_points[..., None, None, :]
        key_points = torch.cat(
            [
                key_points,
                key_points.new_full(key_points.shape[:-1]+(1,), fill_value=self.ground_height),
            ],
            dim=-1,
        )
        fix_height = key_points.new_tensor(self.fix_height)
        height_offset = key_points.new_zeros([len(fix_height), 2])
        height_offset = torch.cat([height_offset, fix_height[:,None]], dim=-1)
        key_points = key_points + height_offset[None, None, None, :, None]
        key_points = key_points.flatten(2, 4)
        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        temp_key_points_list = []
        for i, t_time in enumerate(temp_timestamps):
            temp_key_points = key_points
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    # @staticmethod
    def anchor_projection(
        self,
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            dst_anchor = anchor.clone()
            bs, num_anchor, _ = anchor.shape
            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(1, 2)
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            dst_anchor = (
                torch.matmul(
                    T_src2dst[..., :2, :2], dst_anchor[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :2, 3]
            )

            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(2, 3)
            dst_anchors.append(dst_anchor)
        return dst_anchors


@HEADS.register_module()
class DriveStyleMotionPlanningHead(BaseModule):
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
        
        diff_operation_order=None,
        diff_noise_operation_order=None,
        diff_input_dim=13, 
        diff_hidden_dim=256, 
        diff_T=10, 
        diff_betas=None,
        self_attn_model=None,
        diff_refine_layer=None,
        diff_refine_noise_layer=None,
        traj_pooler_layer=None,
        use_tp=None,
    ):
        super(DriveStyleMotionPlanningHead, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        
        self.ego_fut_cmd = 3
        self.diff_T = diff_T
        self.diff_beta_start = 1e-4
        self.diff_beta_end = 0.02
        self.diff_betas = torch.linspace(self.diff_beta_start, self.diff_beta_end, self.diff_T)  # 噪声调度
        self.diff_alphas = 1 - self.diff_betas
        self.diff_alphas_cumprod = torch.cumprod(self.diff_alphas, dim=0)
        self.diff_input_dim=diff_input_dim
        self.diff_hidden_dim=diff_hidden_dim     
        self.diff_operation_order = diff_operation_order
        self.diff_noise_operation_order = diff_noise_operation_order   

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
            
            "agent_cross_gnn": [graph_model, ATTENTION],
            "map_cross_gnn": [cross_graph_model, ATTENTION],
            "anchor_cross_gnn": [cross_graph_model, ATTENTION],
            "diff_refine": [diff_refine_layer, PLUGIN_LAYERS],
            "diff_refine_noise": [diff_refine_noise_layer, PLUGIN_LAYERS],
            "traj_pooler": [traj_pooler_layer, PLUGIN_LAYERS],
            "self_attn": [self_attn_model, ATTENTION],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        
        self.diff_layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.diff_operation_order
            ]
        )
        
        self.diff_noise_layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.diff_noise_operation_order
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
        plan_anchor = np.load(plan_anchor)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )
        
        self.plan_pos_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1,768),
            Linear(embed_dims, embed_dims),
        )
        

        self.num_det = num_det
        self.num_map = num_map


        self.use_tp = use_tp
        if self.use_tp is not None:
            self.tp_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 2)
            )
        self.kps_generator = TrajSparsePoint3DKeyPointsGenerator(embed_dims=embed_dims, 
                                                                num_sample=ego_fut_ts,
                                                                fix_height=(0, 0.5, -0.5, 1, -1),
                                                                ground_height=-1.84023,)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        #self.time_mlp = nn.Sequential(
        #    SinusoidalPosEmb(embed_dims),
        #    nn.Linear(embed_dims, embed_dims * 4),
        #    nn.Mish(),
        #    nn.Linear(embed_dims * 4, embed_dims),
        #)
        self.diff_T = diff_T  # 最大时间步
        
        # 设定 beta 和 alpha
        if diff_betas is None:
            diff_betas = torch.linspace(0.0001, 0.02, self.diff_T)  # 线性增长的 beta 序列
        self.diff_betas = diff_betas
        self.diff_betas = 1.0 - diff_betas
        self.alphas_cumprod = torch.cumprod(self.diff_alphas, dim=0)  # 计算累积 alpha
        
        # 轨迹去噪网络
        self.diffusion_denoiser1 = nn.Sequential(
            nn.Linear(self.diff_input_dim, 64),  # 13 -> 64
            nn.ReLU(),
            nn.Linear(64, self.diff_hidden_dim * 6),  # 64 -> 6 * 256
            nn.ReLU(),
            nn.Unflatten(1, (6, self.diff_hidden_dim))  # [2, 6 * 256] -> [2, 6, 256]
        )
        self.diffusion_denoiser2 = nn.Sequential(
            nn.Linear(self.diff_hidden_dim, 64),  # 256 -> 64
            nn.ReLU(),
            nn.Linear(64, 1),  # 64 -> 1（逐步降维）
            nn.ReLU(),
            nn.Flatten(),  # [2, 6, 1] -> [2, 6]
            nn.Linear(6, 12)  # [2, 6] -> [2, 12]
        )


    def diffusion_denoiser(self,reg_target_t):
        predicted_noise = self.diffusion_denoiser1(reg_target_t)#[2, 6, 256]
        
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "norm" or op == "ffn":
                predicted_noise = self.layers[i](predicted_noise)
            elif op == "cross_gnn":
                predicted_noise = self.layers[i](
                    predicted_noise,
                    predicted_noise,
                    predicted_noise,
                )
                

        predicted_noise = self.diffusion_denoiser2(predicted_noise)#torch.Size([2, 12])
        return predicted_noise

    def diffusion_forward_process(self, reg_target, t):
        """前向过程：加噪"""

        noise = torch.randn_like(reg_target)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1).to(reg_target.device)
        reg_target_t = torch.sqrt(alpha_cumprod_t) * reg_target + torch.sqrt(1 - alpha_cumprod_t) * noise
        return reg_target_t, noise

    def diffusion_reverse_process(self, reg_target_t, t):
        """反向过程：去噪"""
        t_embed = torch.sin(t.float().unsqueeze(1) * 2 * torch.pi / self.diff_T).to(reg_target_t.device)
        #import pdb;pdb.set_trace()
        reg_target_t = reg_target_t.view(reg_target_t.size(0), -1)  # 展平输入
        reg_target_t = torch.cat([reg_target_t, t_embed], dim=1)  # 结合时间步信息
        #import pdb;pdb.set_trace()
        predicted_noise = self.diffusion_denoiser(reg_target_t)

        predicted_noise = predicted_noise.view(reg_target_t.size(0), int(reg_target_t.size(1)/2), - 1)
        return predicted_noise

    def diffusion_loss_fn(self, predicted_noise,noise):  #reg_target [torch.Size([1, 1, 6, 2])]
        """计算损失函数"""
        #import pdb;pdb.set_trace()
        noise = noise.reshape(noise.shape[0]*noise.shape[1],
                            1, noise.shape[2], noise.shape[3])  # squeeze 掉第 2 维的 1
        predicted_noise = torch.stack(predicted_noise)
        loss = F.mse_loss(predicted_noise, noise)  # 计算噪声的 MSE 损失
        return loss

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

    # 碰撞检测
    def check_collision(self, trajectories, agents, safe_distance=2.0):
        #safe_trajectories = []
        #for traj in trajectories:
        safe = True
        for agent in agents:
            # import pdb; pdb.set_trace()
            min_dist = torch.min(torch.norm(trajectories - agent[:6], dim=-1))
            if min_dist >= safe_distance:
                safe = False
                break
        if safe:
            return True
        else:
            return False

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
    def diff_graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        # import ipdb;ipdb.set_trace()
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.diff_layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
    )
    
    def diff_noise_graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        # import ipdb;ipdb.set_trace()
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.diff_noise_layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
    )    
    
    def normalize_ego_fut_trajs(self, gt_ego_fut_trajs):
        # bs, ego_fut_ts, _ = gt_ego_fut_trajs.shape
        odo_info_fut_x = gt_ego_fut_trajs[..., 0:1]
        odo_info_fut_y = gt_ego_fut_trajs[..., 1:2]

        odo_info_fut_x = odo_info_fut_x / 3
        odo_info_fut_x = odo_info_fut_x.clamp(-1, 1)
        odo_info_fut_y = (odo_info_fut_y+0.5) / 8.1
        odo_info_fut_y = odo_info_fut_y.clamp(0, 1)
        odo_info_fut_y = odo_info_fut_y * 2 - 1
        odo_info_fut = torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
        # odo_info_fut = odo_info_fut.reshape(-1,self.ego_fut_ts, 2)
        return odo_info_fut

    def denormalize_ego_fut_trajs(self, noisy_traj_points):
        # bs, ego_fut_ts, _ = noisy_traj_points.shape
        odo_info_fut_x = noisy_traj_points[..., 0:1]
        odo_info_fut_y = noisy_traj_points[..., 1:2]

        odo_info_fut_x = odo_info_fut_x * 3
        # odo_info_fut_x = odo_info_fut_x.clamp(-1, 1)
        odo_info_fut_y = (odo_info_fut_y+1) / 2 * 8.1 - 0.5
        # odo_info_fut_y = odo_info_fut_y.clamp(-1, 1)
        odo_info_fut = torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
        return odo_info_fut
    def split_tensor_to_list(self,predicted_noise_diff_plan_reg,predicted_noise_list):
        """将形状为 [6, 1, 6, 6, 2] 的张量拆分为 36 个 [1, 6, 2] 的张量列表"""
        #import pdb;pdb.set_trace()
        return predicted_noise_diff_plan_reg.reshape(len(predicted_noise_list)//self.ego_fut_cmd, predicted_noise_list[0].size(0), predicted_noise_list[0].size(2), predicted_noise_list[0].size(3)).unbind(0)
    
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

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        plan_anchor = torch.tile(
            self.plan_anchor[None], (bs, 1, 1, 1, 1)
        )

        # =========== mode query init ===========
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)

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
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2)

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
                )
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)

        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)



        noise_list = []
        predicted_noise_list = []
        for idx in range(plan_reg.size(0) * plan_reg.size(2)):
            i = idx // plan_reg.size(2)  # 对应原来的 i
            j = idx % plan_reg.size(2)   # 对应原来的 j

            plan_reg_item = plan_reg[i, :, j, :, :]  # shape: [1, 6, 2]
            plan_reg_item = plan_reg_item.unsqueeze(0)        # shape: [1, 1, 6, 2]

            t = torch.randint(0, self.diff_T, (plan_reg_item.size(0),), device=plan_reg.device)  # 随机时间步
            reg_target_t, noise = self.diffusion_forward_process(plan_reg_item, t)  # 加噪（预测的多模态轨迹）
            #import pdb;pdb.set_trace()
            predicted_noise = self.diffusion_reverse_process(reg_target_t, t)  # 预测噪声（预测的多模态轨迹）
            predicted_noise = predicted_noise.unsqueeze(1)
            
            #import pdb;pdb.set_trace()
            noise_list.append(noise)
            predicted_noise_list.append(predicted_noise)

        predicted_noise_tensor=torch.stack(predicted_noise_list).squeeze(1)#torch.Size([36, 6, 2])#预测噪声（预测的多模态轨迹）
        # import pdb;pdb.set_trace()
        predicted_noise_tensor_cmd = predicted_noise_tensor.view(bs,
            self.ego_fut_cmd,
            self.ego_fut_mode,
            self.ego_fut_ts,
            predicted_noise_tensor.shape[3])#（预测的多模态轨迹）#6,3,6,6,2
        #import pdb;pdb.set_trace()

        
        #给预测的噪声加条件---------------------------------------------------------bencheng
        bs_indices = torch.arange(bs, device=plan_query.device)#tensor([0, 1, 2, 3, 4, 5], device='cuda:0')
        #import pdb;pdb.set_trace()
        cmd = metas['gt_ego_fut_cmd'].argmax(dim=-1)#tensor([0, 0, 0, 0, 0, 0], device='cuda:0')

        # plan_anchor torch.Size([6, 6, 1, 6, 2])
        cmd_plan_anchor = plan_anchor[bs_indices, cmd] #torch.Size([6, 1, 6, 2])
        predicted_noise_tensor_cmd = predicted_noise_tensor_cmd[bs_indices, cmd]#（预测的多模态轨迹）
        #import pdb;pdb.set_trace()
        zeros_cat = torch.zeros(bs, 6, 1, 2, device=plan_query.device)#torch.Size([6, 1, 1, 2])
        #import pdb;pdb.set_trace()
        cmd_plan_anchor = torch.cat([zeros_cat,cmd_plan_anchor], dim=2)#torch.Size([6, 1, 7, 2])
        tgt_cmd_plan_anchor = cmd_plan_anchor[:,:,1:,:] - cmd_plan_anchor[:,:,:-1,:]#torch.Size([1, 6, 6, 2])
        odo_info_fut = self.normalize_ego_fut_trajs(tgt_cmd_plan_anchor)#torch.Size([1, 6, 6, 2])
        # import pdb; pdb.set_trace()
        
        odo_info_fut = odo_info_fut.view(bs*self.ego_fut_mode ,self.ego_fut_ts,2)

        timesteps = torch.randint(
            0, 40,
            (bs,), device=plan_query.device
        )# TODO, only bs timesteps


        repeat_timesteps = timesteps.repeat_interleave(self.ego_fut_mode)
        #import pdb;pdb.set_trace()
        #odo_info_fut = odo_info_fut.repeat(6, 1, 1)
        #import pdb;pdb.set_trace()
        noise = torch.randn(odo_info_fut.shape, device=plan_query.device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(original_samples=odo_info_fut,noise=noise,timesteps=repeat_timesteps).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denormalize_ego_fut_trajs(noisy_traj_points) # bs, ego_fut_ts, 2


        diff_plan_reg = noisy_traj_points
        #import pdb;pdb.set_trace()
        traj_pos_embed = gen_sineembed_for_position(diff_plan_reg,hidden_dim=128)
        predicted_noise_traj_pos_embed = gen_sineembed_for_position(predicted_noise_tensor_cmd,hidden_dim=128)#（预测的多模态轨迹）
        
        traj_pos_embed = traj_pos_embed.flatten(-2)
        predicted_noise_traj_pos_embed = predicted_noise_traj_pos_embed.flatten(-2)#（预测的多模态轨迹）
        
        traj_feature = self.plan_pos_encoder(traj_pos_embed)
        predicted_noise_traj_feature = self.plan_pos_encoder(predicted_noise_traj_pos_embed)#（预测的多模态轨迹）



        traj_feature = traj_feature.view(bs, self.ego_fut_mode, -1)
        predicted_noise_traj_feature = predicted_noise_traj_feature.view(bs,self.ego_fut_mode,-1)#（预测的多模态轨迹）
        #import pdb;pdb.set_trace()

        # plan_query torch.Size([6, 1, 6, 256])
        plan_nav_query = plan_query.squeeze(1)
        
        mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),  # 可以加也可以不加
        ).to(plan_query.device)
        plan_nav_query = mlp(plan_nav_query)

        
        # import pdb; pdb.set_trace()
        
        plan_nav_query = plan_nav_query.view(bs,3,6,-1)#torch.Size([6, 6, 6, 256])

        cmd_plan_nav_query = plan_nav_query[bs_indices, cmd]#cmd_plan_nav_query torch.Size([6, 6, 256])
        # import pdb; pdb.set_trace()
        #time_embed = self.time_mlp(repeat_timesteps)
        # time_embed = self.time_mlp(timesteps)#torch.Size([6, 256])
        #time_embed = time_embed.view(bs,self.ego_fut_mode,-1)

        diff_planning_prediction = []
        diff_planning_classification = []
        repeat_ego_anchor_embed = ego_anchor_embed.repeat(1,self.ego_fut_mode,1)
        for i, op in enumerate(self.diff_operation_order):
            if self.diff_layers[i] is None:
                continue

            elif op == "traj_pooler":
                if len(diff_plan_reg.shape) != 3:
                    diff_plan_reg = diff_plan_reg[:,:,-self.ego_fut_mode:,].flatten(0,2)#torch.Size([36, 6, 2])
                #import pdb;pdb.set_trace()
                traj_feature = self.diff_layers[i](traj_feature,diff_plan_reg,metas,feature_maps,modal_num=self.ego_fut_mode)
                #import pdb;pdb.set_trace()
            elif op == "self_attn":
                #import pdb;pdb.set_trace()
                traj_feature = self.diff_layers[i](
                    traj_feature,
                    traj_feature,
                    traj_feature,
                )
                # print(traj_feature.shape)#torch.Size([6, 6, 256])
            elif op == "agent_cross_gnn":#3
                #import pdb;pdb.set_trace()
                traj_feature = self.diff_graph_model(
                    i,
                    traj_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=repeat_ego_anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "map_cross_gnn":
                #import pdb;pdb.set_trace()
                traj_feature = self.diff_layers[i](
                    traj_feature,
                    map_instance_feature_selected,
                    map_instance_feature_selected,
                    query_pos=repeat_ego_anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "anchor_cross_gnn":
                # import pdb;pdb.set_trace()
                traj_feature = self.diff_layers[i](
                    traj_feature,
                    key=cmd_plan_nav_query,
                    value=cmd_plan_nav_query,
                )
            elif op == "norm" or op == "ffn":
                #import pdb;pdb.set_trace()
                traj_feature = self.diff_layers[i](traj_feature)
            elif op == "diff_refine":
                diff_plan_reg, diff_plan_cls = self.diff_layers[i](
                    traj_feature,
                )
        #import pdb;pdb.set_trace()
        for i, op in enumerate(self.diff_noise_operation_order):
            if self.diff_noise_layers[i] is None:
                continue

            elif op == "traj_pooler":
                predicted_noise_traj_feature = self.diff_noise_layers[i](predicted_noise_traj_feature,
                        predicted_noise_tensor_cmd.view(bs*predicted_noise_tensor_cmd.shape[1],
                                predicted_noise_tensor_cmd.shape[2],
                                predicted_noise_tensor_cmd.shape[3]),
                        metas,
                        feature_maps,
                        modal_num=self.ego_fut_mode
                    )#（预测的多模态轨迹）
            elif op == "self_attn":
                predicted_noise_traj_feature = self.diff_noise_layers[i](
                    predicted_noise_traj_feature,
                    predicted_noise_traj_feature,
                    predicted_noise_traj_feature,
                )#（预测的多模态轨迹）
            elif op == "agent_cross_gnn":#3
                predicted_noise_traj_feature = self.diff_noise_graph_model(
                    i,
                    predicted_noise_traj_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=repeat_ego_anchor_embed,
                    key_pos=anchor_embed_selected,
                )#（预测的多模态轨迹）
            elif op == "map_cross_gnn":
                predicted_noise_traj_feature = self.diff_noise_layers[i](
                    predicted_noise_traj_feature,
                    map_instance_feature_selected,
                    map_instance_feature_selected,
                    query_pos=repeat_ego_anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )#（预测的多模态轨迹）
            elif op == "anchor_cross_gnn":
                predicted_noise_traj_feature = self.diff_noise_layers[i](
                    predicted_noise_traj_feature,
                    key=cmd_plan_nav_query,
                    value=cmd_plan_nav_query,
                )#（预测的多模态轨迹）
            elif op == "norm" or op == "ffn":
                predicted_noise_traj_feature = self.diff_noise_layers[i](predicted_noise_traj_feature)#（预测的多模态轨迹）
            elif op == "diff_refine_noise":
                predicted_noise_diff_plan_reg = self.diff_noise_layers[i](
                    predicted_noise_traj_feature,
                )    #（预测的多模态轨迹）

        diff_planning_prediction.append(diff_plan_reg)
        diff_planning_classification.append(diff_plan_cls)
        noise_list_tensor = torch.stack(noise_list).reshape(bs,self.ego_fut_cmd,self.ego_fut_mode,6,2)
        noise_list_tensor = noise_list_tensor[bs_indices, cmd]
        #import pdb;pdb.set_trace()
        # predicted_noise_list.append(predicted_noise_diff_plan_reg)

        predicted_noise_diff_plan_reg = self.split_tensor_to_list(predicted_noise_diff_plan_reg,predicted_noise_list)#（预测的多模态轨迹）


        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
   
        
        planning_output = {
            "diff_classification": diff_planning_classification, #1216#diff_classification
            "diff_prediction": diff_planning_prediction,#1216 62预测的多模态轨迹）
            "classification": planning_classification, #1216
            "prediction": planning_prediction,#1216 62
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
            "noise": noise_list_tensor,#（随机噪声）
            "predicted_noise":predicted_noise_diff_plan_reg #（
            # "predicted_noise":predicted_noise_list 
        }
             # import pdb;pdb.set_trace()

        return motion_output, planning_output
    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        # diff_planning_loss = self.diff_loss_planning(planning_model_outs, data)
        # loss.update(diff_planning_loss)
        
        
        diff_PPO_loss_planning = self.diff_PPO_loss_planning(planning_model_outs, data)
        loss.update(diff_PPO_loss_planning)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
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

    
    def generate_safe_trajectories(self, reg_target, gt_agent_fut_trajs, num_trajectories, safe_distance=2.0):
        #sampled_trajectories = [self.reverse_process_sample(reg_target.shape, reg_target.device) for _ in range(num_trajectories)]#10,3,6,2
        # import pdb;pdb.set_trace()
        safe_trajectories = [[] for _ in range(reg_target.shape[0])]
        for i in range(num_trajectories):
            sampled_trajectories = self.reverse_process_sample(reg_target)#加
            #import pdb;pdb.set_trace()
            for i_bs in range(reg_target.shape[0]):
                if self.check_collision(sampled_trajectories[i_bs], gt_agent_fut_trajs[i_bs], safe_distance):
                    safe_trajectories[i_bs].append(sampled_trajectories[i_bs])
        #import pdb;pdb.set_trace()
        return safe_trajectories
    
    def generate_safe_ppo_trajectories(self, reg_target, gt_agent_fut_trajs, num_trajectories, safe_distance, predicted_noise, noise):
        #sampled_trajectories = [self.reverse_process_sample(reg_target.shape, reg_target.device) for _ in range(num_trajectories)]#10,3,6,2
        import pdb;pdb.set_trace()
        safe_trajectories = [[] for _ in range(reg_target.shape[0])]
        for i in range(num_trajectories):
            sampled_trajectories = self.reverse_ppo_process_sample(reg_target,predicted_noise, noise)#加
            #import pdb;pdb.set_trace()
            for i_bs in range(reg_target.shape[0]):
                if self.check_collision(sampled_trajectories[i_bs], gt_agent_fut_trajs[i_bs], safe_distance):
                    safe_trajectories[i_bs].append(sampled_trajectories[i_bs])
        #import pdb;pdb.set_trace()
        return safe_trajectories
    def reverse_ppo_process_sample(self,reg_target,predicted_noise, noise):
        # 初始化随机噪声
        #import pdb;pdb.set_trace()
        bs, num_gt , ts, num_pos = reg_target.shape
        reg_target_t = torch.randn([bs, ts, num_pos], device=reg_target.device) #初始化噪声
        reg_target_t=reg_target.squeeze(1) + reg_target_t #GT+噪声
        # 逐步去噪
        for t in range(self.diff_T - 1, -1, -1):
            t_tensor = torch.full((reg_target.shape[0],), t, device=reg_target.device, dtype=torch.long)  # 当前时间步
            predicted_noise = self.diffusion_reverse_process(reg_target_t, t_tensor)  # 预测噪声
            alpha_t = self.diff_alphas_cumprod[t]
            alpha_t_prev = self.diff_alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

            # 计算去噪后的 x_{t-1}
            beta_t = self.diff_betas[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_beta_t = torch.sqrt(beta_t)

            # 更新 reg_target_t
            if t > 0:
                noise = torch.randn_like(reg_target_t)  # 添加随机噪声
            else:
                noise = torch.zeros_like(reg_target_t)  # 最后一步不加噪声
            #import pdb; pdb.set_trace()
            reg_target_t = (reg_target_t - (1 - alpha_t) / sqrt_alpha_t * predicted_noise) / torch.sqrt(alpha_t)
            reg_target_t = reg_target_t * sqrt_alpha_t_prev + sqrt_beta_t * noise
        #import pdb;pdb.set_trace()
        return reg_target_t
    
    def reverse_process_sample(self,reg_target):
        # 初始化随机噪声
        #import pdb;pdb.set_trace()
        bs, num_gt , ts, num_pos = reg_target.shape
        reg_target_t = torch.randn([bs, ts, num_pos], device=reg_target.device) #初始化噪声
        reg_target_t=reg_target.squeeze(1) + reg_target_t #GT+噪声
        # 逐步去噪
        for t in range(self.diff_T - 1, -1, -1):
            t_tensor = torch.full((reg_target.shape[0],), t, device=reg_target.device, dtype=torch.long)  # 当前时间步
            predicted_noise = self.diffusion_reverse_process(reg_target_t, t_tensor)  # 预测噪声
            alpha_t = self.diff_alphas_cumprod[t]
            alpha_t_prev = self.diff_alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

            # 计算去噪后的 x_{t-1}
            beta_t = self.diff_betas[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_beta_t = torch.sqrt(beta_t)

            # 更新 reg_target_t
            if t > 0:
                noise = torch.randn_like(reg_target_t)  # 添加随机噪声
            else:
                noise = torch.zeros_like(reg_target_t)  # 最后一步不加噪声
            #import pdb; pdb.set_trace()
            reg_target_t = (reg_target_t - (1 - alpha_t) / sqrt_alpha_t * predicted_noise) / torch.sqrt(alpha_t)
            reg_target_t = reg_target_t * sqrt_alpha_t_prev + sqrt_beta_t * noise
        #import pdb;pdb.set_trace()
        return reg_target_t
    
    def pad_sublist(self, sublist, max_length, device):
    # 计算需要填充的数量
        pad_size = max_length - len(sublist)
        if pad_size > 0:
            # 创建一个填充的 Tensor，形状为 [pad_size, 6, 2]
            padding = torch.zeros(pad_size, 6, 2)
            #import pdb;pdb.set_trace()
            #if pad_size != max_length:
            padding = padding.to(device)
            if pad_size != max_length:
                sublist = torch.cat([torch.stack(sublist), padding], dim=0)
            else:
                sublist = padding
        else:
            sublist = torch.stack(sublist)
        return sublist
    
    
    def compute_reward(self, reg_all, data):
        """
        reg_all: Tensor [B, M, T, 2] - 多模态预测轨迹
        data: 包含 'gt_agent_fut_trajs'（list）和 'gt_agent_fut_masks'（list）
        return:
            - safety_reward: [B]
            - map_reward: [B]（dummy 设为1）
        """
        batch_size, num_modes, T, _ = reg_all.shape
        device = reg_all.device

        safety_reward = torch.ones(batch_size, device=device)
        map_reward = torch.ones(batch_size, device=device)  # dummy 值

        safe_distance = 2.0
        other_agents = data['gt_agent_fut_trajs']  # list of [A, T, 2]
        agent_masks = data['gt_agent_fut_masks']   # list of [A, T]

        for b in range(batch_size):
            collision_penalty = 0.0
            if len(other_agents[b]) == 0:
                continue
            for m in range(num_modes):
                ego_traj = reg_all[b, m]  # [T, 2]
                for a in range(other_agents[b].shape[0]):
                    other_traj = other_agents[b][a]  # [T, 2]
                    valid_mask = agent_masks[b][a] > 0  # [T]
                    if valid_mask.sum() == 0:
                        continue
                    # 有效帧距离
                    # import pdb; pdb.set_trace()
                    
                    dists = torch.norm(ego_traj - other_traj[:6], dim=1)  # [T]
                    # import pdb; pdb.set_trace()
                    collision = (dists < safe_distance) & valid_mask[:6]
                    if collision.any():
                        collision_penalty += 1.0
            safety_reward[b] = 1.0 - min(1.0, collision_penalty / num_modes)

        return safety_reward, map_reward


    # @force_fp32(apply_to=("model_outs"))
    def diff_PPO_loss_planning(self, model_outs, data):
        cls_scores = model_outs["diff_classification"]
        reg_preds = model_outs["diff_prediction"]
        predicted_noise = model_outs["predicted_noise"]
        noise = model_outs["noise"]
        output = {}

        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            # === Step 1: 多模轨迹 -> PPO-style Reward ===
            # === 多样性奖励 ===
            with torch.no_grad():
                reg_all = reg[decoder_idx]  # [B, M, T, 2]
                batch_size, num_modes, T, _ = reg_all.shape
                diversity_reward = torch.zeros(batch_size, device=reg_all.device)

                for i in range(num_modes):
                    for j in range(i + 1, num_modes):
                        dist = F.pairwise_distance(reg_all[:, i], reg_all[:, j], p=2).mean(dim=-1)  # [B]
                        diversity_reward += dist

                diversity_reward = diversity_reward / (num_modes * (num_modes - 1) / 2)  # 平均奖励



                safety_reward, map_reward = self.compute_reward(reg_all, data)  # [B], [B]
                total_reward = (
                    0.5 * diversity_reward + 
                    0.3 * safety_reward +
                    0.2 * map_reward
                )
            

            # === Step 2: Diffusion Loss（标准重建） ===
            diffusion_loss = self.diffusion_loss_fn(predicted_noise, noise)

            # === Step 3: Imitation Learning 监督（采样最优轨迹） ===
            cls, cls_target, cls_weight, reg_pred, reg_target, reg_weight = self.planning_sampler.sample(
                cls, reg, data['gt_ego_fut_trajs'], data['gt_ego_fut_masks'], data
            )
            multi_reg_targets = self.generate_safe_trajectories(reg_target, data['gt_agent_fut_trajs'], num_trajectories=10,safe_distance=2.0)
            # 分类损失
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            # 回归损失
            reg_weight = reg_weight.flatten(end_dim=1).unsqueeze(-1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_loss_target = self.plan_loss_reg(reg_pred, reg_target, weight=reg_weight)

            max_length = 10
            list_lengths = [len(sublist) for sublist in multi_reg_targets]
            padded_multi_reg_targets = list(map(lambda x: self.pad_sublist(x, max_length, reg_target.device), multi_reg_targets))
            padded_multi_reg_targets = torch.stack(padded_multi_reg_targets)
            
            mask_multi_target = torch.ones((len(list_lengths), max_length), device=padded_multi_reg_targets.device)

            indices = torch.arange(max_length)


            list_tensor = torch.tensor(list_lengths).unsqueeze(1)

            mask_multi_target[indices >= list_tensor] = 0
            reg_loss_multi_safe_target = 0
            for i_loss in range(max_length):
                i_mask = mask_multi_target[:,i_loss].unsqueeze(1).repeat(1, 6).unsqueeze(2)
                reg_loss_multi_safe_target += self.plan_loss_reg(reg_pred, padded_multi_reg_targets[:,i_loss,...], weight=reg_weight*i_mask)

            reg_loss=(reg_loss_target*0.9+reg_loss_multi_safe_target*0.1)

            diffusion_loss = diffusion_loss * (1.0 + total_reward.mean())
            # import pdb; pdb.set_trace()
            output.update({
                f"diff_PPO_planning_loss_cls_{decoder_idx}": cls_loss*0.01,
                f"diff_PPO_planning_loss_reg_{decoder_idx}": reg_loss*0.01,
                f"diff_PPO_loss_{decoder_idx}": diffusion_loss*0.01,
                f"diff_PPO_reward_total_{decoder_idx}": total_reward.mean().detach()*0.01,
            })

        return output





    
    @force_fp32(apply_to=("model_outs"))
    def diff_loss_planning(self, model_outs, data):
        cls_scores = model_outs["diff_classification"]  # [1, 2, 1, 6]
        reg_preds = model_outs["diff_prediction"]       # [1, 2, 1, 6, 6, 2]
        predicted_noise = model_outs["predicted_noise"] 
        noise = model_outs["noise"] 
        output = {}
        #import pdb;pdb.set_trace()

        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            #import pdb;pdb.set_trace()
            (
                cls,                    # [torch.Size([1, 1, 6])]
                cls_target,             # [torch.Size([1, 1])]
                cls_weight,             # [torch.Size([1, 1])]
                reg_pred,               # [torch.Size([1, 1, 6, 2])]
                reg_target,             # [torch.Size([1, 1, 6, 2])]
                reg_weight,             # [torch.Size([1, 1, 6])]
            ) = self.planning_sampler.sample(
                cls, reg, data['gt_ego_fut_trajs'], data['gt_ego_fut_masks'], data
            )

            # 轨迹生成部分（使用扩散模型生成轨迹）
            #import pdb;pdb.set_trace()
            diffusion_loss = self.diffusion_loss_fn(predicted_noise,noise)
            #import pdb;pdb.set_trace()
            # fake_gt_trajectories = 
            #import pdb; pdb.set_trace()
            # 得到去噪后的虚拟gt 轨迹 shape = [bs, 6, 2]
            
            multi_reg_targets = self.generate_safe_trajectories(reg_target, data['gt_agent_fut_trajs'], num_trajectories=10,safe_distance=2.0)



            # 分类损失
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            # 回归损失
            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_loss_target = self.plan_loss_reg(reg_pred, reg_target, weight=reg_weight)
            #diff_reg_loss_target = self.plan_loss_reg(diff_reg_pred, reg_target, weight=reg_weight)
            max_length = 10
            list_lengths = [len(sublist) for sublist in multi_reg_targets]
            padded_multi_reg_targets = list(map(lambda x: self.pad_sublist(x, max_length, reg_target.device), multi_reg_targets))
            padded_multi_reg_targets = torch.stack(padded_multi_reg_targets)
            
            mask_multi_target = torch.ones((len(list_lengths), max_length), device=padded_multi_reg_targets.device)

            # 生成一个长度为 max_length 的索引张量
            indices = torch.arange(max_length)

            # 将 list 转换为张量并调整形状以便广播
            list_tensor = torch.tensor(list_lengths).unsqueeze(1)

            # 使用布尔索引将 B 中对应位置的值设置为0
            mask_multi_target[indices >= list_tensor] = 0
            reg_loss_multi_safe_target = 0
            for i_loss in range(max_length):
                i_mask = mask_multi_target[:,i_loss].unsqueeze(1).repeat(1, 6).unsqueeze(2)
                reg_loss_multi_safe_target += self.plan_loss_reg(reg_pred, padded_multi_reg_targets[:,i_loss,...], weight=reg_weight*i_mask)
            #import pdb; pdb.set_trace()
            #reg_loss_multi_safe_target = self.plan_loss_reg(reg_pred, multi_reg_targets, weight=reg_weight)
            reg_loss=reg_loss_target*0.9+reg_loss_multi_safe_target*0.1
            # 状态损失
            #status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"diff_planning_loss_cls_{decoder_idx}": cls_loss,
                    f"diff_planning_loss_reg_{decoder_idx}": reg_loss,
                    f"diffusion_loss_{decoder_idx}": diffusion_loss,
                }
            )

        return output



    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )

            cls = cls.flatten(end_dim=1) # [6,1,6] , [6,6]
            cls_target = cls_target.flatten(end_dim=1) # [6,1], [6]
            cls_weight = cls_weight.flatten(end_dim=1) # [6,1], [6]
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

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