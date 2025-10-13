import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)


@PLUGIN_LAYERS.register_module()
class FocalADMotionFeatureRefine(nn.Module):
    def __init__(
        self,
        d_node=64,
        d_edge=16,
        d_model=256,
        topk=5,
        num_heads=4,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # 指定 device 与 dtype
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype if dtype is not None else torch.float32

        self.d_node, self.d_edge = d_node, d_edge
        self.d_model, self.topk = d_model, topk
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # --- 编码层 ---
        self.node_encoder = nn.Linear(8, d_node)
        self.edge_encoder = nn.Linear(3, d_edge)

        # --- EGAT 投影 ---
        self.W_q_egat = nn.Linear(d_node, d_model)
        self.W_kv_egat = nn.Linear(d_node + d_edge + d_node, d_model * 2)


        self.ego_fuse = nn.Sequential(
            nn.Linear(d_node + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(d_node + d_edge + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, (1+6*d_model)) # 6为了和motion query匹配， +1 为了多得到个分数
        )

        # 将模块参数统一放到指定的 device 与 dtype
        self.to(self.device)
        self.to(self.dtype)

    def forward(self, node_raw, edge_raw, ego_id: int):
        # 自动构造 edge_index: 每个 batch 以 ego 为中心连接其他邻居
        node_raw = node_raw.to(device=self.device, dtype=self.dtype)
        edge_raw = edge_raw.to(device=self.device, dtype=self.dtype)

        B, N, _ = node_raw.shape
        E = N - 1
        device = node_raw.device

        # 构造 edge_index: shape (B, E, 2) 每条边是 (src, dst)
        dst = torch.arange(N, device=device)
        dst = dst[dst != ego_id]  # (E,)
        src = torch.full_like(dst, fill_value=ego_id)
        single_edge_index = torch.stack([src, dst], dim=-1)  # (E, 2)
        edge_index = single_edge_index.unsqueeze(0).repeat(B, 1, 1)  # (B, E, 2)

        h_ego_prime_list, selected_feat_list, topk_idx_list, alpha_list =[], [], [], []

        for b in range(B):
            node_raw_b = node_raw[b]  # (N, D)
            edge_raw_b = edge_raw[b]  # (E, D_edge)
            edge_index_b = edge_index[b]  # (E, 2)

            node_feat = self.node_encoder(node_raw_b)  # (N, D_node)
            edge_feat = self.edge_encoder(edge_raw_b)  # (E, D_edge)

            ego_feat = node_feat[ego_id]  # (D_node)
            neighbor_idx = edge_index_b[:, 1]  # (E,)
            neighbor_feat = node_feat[neighbor_idx]  # (E, D_node)

            edge_input = torch.cat([
                ego_feat.unsqueeze(0).expand(E, -1),  # (E, D_node)
                edge_feat,                           # (E, D_edge)
                neighbor_feat                        # (E, D_node)
            ], dim=-1)  # (E, D_total)

            # Linear projections
            Q = self.W_q_egat(ego_feat).view(1, self.num_heads * self.head_dim)       # (1, H*D)
            kv = self.W_kv_egat(edge_input).view(E, 2, self.num_heads * self.head_dim)  # (E, 2, H*D)
            K = kv[:, 0]  # (E, H*D)
            V = kv[:, 1]  # (E, H*D)

            # PyTorch MultiheadAttention expects input shape: (L, B, D_model)
            Q_p = Q.unsqueeze(0)  # (1, 1, H*D)
            K_p = K.unsqueeze(1)  # (E, 1, H*D)
            V_p = V.unsqueeze(1)  # (E, 1, H*D)

            # Build MHA module on the fly or as a class member
            mha = MultiheadAttention(embed_dim=self.num_heads * self.head_dim, num_heads=self.num_heads, batch_first=False).to(self.device)

            # Compute attention
            out, _ = mha(Q_p, K_p, V_p)  # out: (1, 1, H*D)
            c_ego = out[0].reshape(-1).to(dtype=self.dtype)

            h_ego_prime = self.ego_fuse(torch.cat([ego_feat, c_ego], dim=-1))

            edge_input2 = torch.cat([
                neighbor_feat,
                edge_feat,
                c_ego.unsqueeze(0).expand(E, -1)
            ], dim=-1)

            score_out= self.score_mlp(edge_input2)
            score = score_out[..., : 1]
            h_enc = score_out[..., 1:].reshape(-1,6,self.d_model)

            topk_val, topk_idx = torch.topk(score.squeeze(1), self.topk, dim=0)
            selected_feat = h_enc[topk_idx]  # (topk, D)
            alpha = F.softmax(topk_val, dim=0)  # (topk,)

            h_ego_prime_list.append(h_ego_prime.unsqueeze(0))
            selected_feat_list.append(selected_feat.unsqueeze(0))  # (1, topk, D)
            topk_idx_list.append(topk_idx.unsqueeze(0))  # (1, topk)
            alpha_list.append(alpha.unsqueeze(0))  # (1, topk)

        h_ego_prime_tensor = torch.cat(h_ego_prime_list, dim=0)
        selected_feat_tensor = torch.cat(selected_feat_list, dim=0)  # (B, topk, D)
        topk_idx_tensor = torch.cat(topk_idx_list, dim=0)  # (B, topk)
        alpha_tensor = torch.cat(alpha_list, dim=0).unsqueeze(-1).unsqueeze(-1) # (B, topk, 1, 1 )
        selected_feat_tensor = selected_feat_tensor * alpha_tensor

        
        # =========== print GPU memory resource ===========
        # print(torch.cuda.memory_allocated() / 1024 / 1024, "MB")

        return h_ego_prime_tensor, selected_feat_tensor, topk_idx_tensor, alpha_tensor


# 外部调用示例
if __name__ == "__main__":
    N, E = 901, 900
    node_raw = torch.randn(6, N, 8)
    edge_raw = torch.randn(6, E, 3)
    ego_id = 0

    # 现在仅需直接创建模型，内部会自动处理设备与 dtype
    model = MotionFeatureRefine()
    h_ego_prime_tensor, selected_feat, topk_idx = model(node_raw, edge_raw, ego_id)
    print("✅ Top-k index:", topk_idx)
