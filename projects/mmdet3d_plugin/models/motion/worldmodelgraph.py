import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 1️⃣ 稀疏图注意力层 =====
class SparseGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn = nn.Parameter(torch.Tensor(num_heads, out_dim * 2))
        nn.init.xavier_uniform_(self.attn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, neighbor_idx):
        """
        x: [B, N, C]
        neighbor_idx: [B, N, K]
        return: [B, N, C]
        """
        B, N, C = x.shape
        K = neighbor_idx.size(-1)
        h = self.fc(x).view(B, N, self.num_heads, -1)  # [B, N, H, C']
        C_h = h.size(-1)

        # Gather邻居特征
        h_i = h.unsqueeze(2)  # [B, N, 1, H, C']
        h_j = torch.gather(
            h.unsqueeze(1).expand(-1, N, -1, -1, -1),
            2,
            neighbor_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.num_heads, C_h)
        )

        # 注意力计算
        e = torch.cat([h_i.expand(-1, -1, K, -1, -1), h_j], dim=-1)
        e = (e * self.attn.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(-1) / (C_h ** 0.5)
        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        # 聚合邻居
        out = torch.einsum("bnkh,bnkhc->bnhc", alpha, h_j).mean(2)
        return out.reshape(B, N, -1)  # [B, N, H*C']


# ===== 2️⃣ 世界模型：Graph-based =====
class WorldModelGraph(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.encoder = nn.Linear(dim, dim)
        self.gat = SparseGATLayer(dim, dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, instance_feature, neighbor_idx):
        x = self.encoder(instance_feature)
        x = self.gat(x, neighbor_idx)
        world_state = self.mlp(x)
        return world_state


# ===== 3️⃣ 主模块 =====
class MotionPlanningHead_PlanWorld_6s(nn.Module):
    def __init__(self, dim=256, num_anchor=900, k=8):
        super().__init__()
        self.num_anchor = num_anchor
        self.k = k
        self.world_model = WorldModelGraph(dim)
        self.graph_proj = nn.Linear(dim, dim)

    def forward(self, instance_feature, anchor_embed, plan_mode_query):
        B, N, C = instance_feature.shape
        assert N > self.num_anchor, f"N={N} must be greater than num_anchor={self.num_anchor}"

        # Step 1: 构建稀疏邻接（K近邻）
        with torch.no_grad():
            dist = torch.cdist(instance_feature, instance_feature)
            _, neighbor_idx = dist.topk(k=self.k, largest=False)

        # Step 2: 图世界建模
        world_state = self.world_model(instance_feature, neighbor_idx)

        # Step 3: 融合生成 plan_query
        plan_query = plan_mode_query + (
            instance_feature[:, self.num_anchor:] +
            anchor_embed[:, self.num_anchor:] +
            self.graph_proj(world_state[:, self.num_anchor:])
        ).unsqueeze(2)
        return plan_query


# ===== 4️⃣ 测试 Demo =====
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, N, C = 6, 901, 256   # 918 = 900 anchors + 18 planning queries
    num_anchor = 900

    instance_feature = torch.randn(B, N, C).to(device) #instance_feature.shape torch.Size([6, 901, 256])
    anchor_embed = torch.randn(B, N, C).to(device) #anchor_embed torch.Size([6, 901, 256])
    plan_mode_query = torch.randn(B, 1, 18, C).to(device)#plan_mode_querytorch.Size([6, 1, 18, 256])
    model = MotionPlanningHead_PlanWorld_6s(dim=C, num_anchor=num_anchor, k=8).to(device)
    plan_query = model(instance_feature, anchor_embed, plan_mode_query)

    print(f"✅ plan_query shape = {plan_query.shape}")
