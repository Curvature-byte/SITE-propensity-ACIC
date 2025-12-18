import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from simi_ite import utils

class PDDMNetwork(nn.Module):
    """
    输入: 两个潜在空间样本 z_i 和 z_j。
    输出: 预测的相似度分数 S_hat (线性输出)。
    """
    def __init__(self, dim_in: int, dim_pddm: int = 32, dim_c: int = 64):
        super(PDDMNetwork, self).__init__()
        self.W_u = nn.Linear(dim_in, dim_pddm)
        self.W_v = nn.Linear(dim_in, dim_pddm)
        self.W_c = nn.Linear(2 * dim_pddm, dim_c)
        
        self.W_s = nn.Linear(dim_c, 1)
        
        self.relu = nn.ReLU()
    # 计算 L2 范数并归一化向量。
    def _normalize_vector(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim = -1, keepdim=True)
        return x / (norm + 1e-10) # 加上 1e-10 防止除以零

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: 第一个潜在空间样本 (batch_size, dim_in)
            z_j: 第二个潜在空间样本 (batch_size, dim_in)
        Returns:
            S_hat: 预测的相似度得分 (batch_size, 1)
        """
        u = torch.abs(z_i - z_j)  # u = |z_i - z_j|
        v = (z_i + z_j) / 2.0 # v = (z_i + z_j) / 2 
        u_norm = self._normalize_vector(u) # u / ||u||_2
        v_norm = self._normalize_vector(v) # v / ||v||_2
        u_1 = self.relu(self.W_u(u_norm))  # u_1 = ReLU(W_u * (u/||u||_2) + b_u)
        v_1 = self.relu(self.W_v(v_norm))  # v_1 = ReLU(W_v * (v/||v||_2) + b_v)
        u_1_norm = self._normalize_vector(u_1) # u_1 / ||u_1||_2
        v_1_norm = self._normalize_vector(v_1) # v_1 / ||v_1||_2
        u_v_concat = torch.cat([u_1_norm, v_1_norm], dim=-1) # [u_1/||u_1||_2, v_1/||v_1||_2]^T
        h = self.relu(self.W_c(u_v_concat)) # h = ReLU(W_c * [...]^T + b_c)
        S_hat = self.W_s(h) # S_hat = W_s * h + b_s
        return S_hat

def compute_pddm_loss(
    pddm_net: PDDMNetwork,
    three_pairs: np.ndarray,
    similarity_ground,
    three_paris_index,
) -> torch.Tensor:
    """
    计算PDDM损失函数
    Args:
        pddm_net: PDDM网络
        three_pairs: 三对样本 (input_dim,) 或 (batch_size, input_dim)
    
    Returns:
        loss: PDDM损失值
    """
    # 预测相似度
    pred_s_kl = pddm_net(three_pairs[2], three_pairs[3]).squeeze()
    pred_s_mn = pddm_net(three_pairs[4], three_pairs[5]).squeeze()
    pred_s_km = pddm_net(three_pairs[2], three_pairs[4]).squeeze()
    pred_s_ik = pddm_net(three_pairs[0], three_pairs[2]).squeeze()
    pred_s_jm = pddm_net(three_pairs[1], three_pairs[4]).squeeze()
    
    # 目标相似度
    target_similarities = utils.get_three_pair_simi(similarity_ground, three_paris_index)
    if isinstance(three_pairs, torch.Tensor):
        target_device = three_pairs.device
    else:
        target_device = next(pddm_net.parameters()).device
    target_similarities = torch.as_tensor(target_similarities, dtype=torch.float32, device=target_device).view(-1)

    # MSE损失
    loss = (
        F.mse_loss(pred_s_kl, target_similarities[0]) +
        F.mse_loss(pred_s_mn, target_similarities[1]) +
        F.mse_loss(pred_s_km, target_similarities[2]) +
        F.mse_loss(pred_s_ik, target_similarities[3]) +
        F.mse_loss(pred_s_jm, target_similarities[4])
    )
    
    return loss / 5.0