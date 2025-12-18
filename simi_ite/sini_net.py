
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pddm_net import PDDMNetwork, compute_pddm_loss


@dataclass
class SITEConfig:
    """Hyper-parameters that mirror the TensorFlow ``FLAGS`` + ``dims`` arrays."""

    dim_input: int
    dim_rep: int
    dim_out: int
    dim_pddm: int
    dim_c: int
    dim_s: int
    n_rep_layers: int = 2
    n_head_layers: int = 2
    activation: str = "elu"  # ``relu`` or ``elu``
    split_output: bool = True
    varsel: bool = False
    batch_norm: bool = False
    normalization: str = "none"  # {"none", "divide"}
    loss_type: str = "l2"  # {"l2", "l1", "log"}
    reweight_sample: bool = False
    rep_weight_decay: bool = True
    dropout_rep: float = 0.1  # dropout probability for representation layers
    dropout_head: float = 0.1  # dropout probability for outcome head
    p_lambda: float = 0.0
    p_mid_point_mini: float = 0.0
    p_pddm: float = 0.0

    def __post_init__(self) -> None:
        if self.activation not in {"elu", "relu"}:
            raise ValueError(f"Unsupported activation: {self.activation}")
        if self.normalization not in {"none", "divide"}:
            raise ValueError(f"Unsupported normalization: {self.normalization}")
        if self.loss_type not in {"l1", "l2", "log"}:
            raise ValueError(f"Unsupported loss: {self.loss_type}")


@dataclass
class SITERegularizationWeights:
    """Runtime scalars that multiply the base ``p_*`` coefficients."""

    weight_decay: float = 1.0
    mid_point: float = 1.0
    pddm: float = 1.0


@dataclass
class SITEForwardOutputs:
    """Convenience container so the loss function receives consistent tensors."""

    y_pred: torch.Tensor
    representation: torch.Tensor
    pair_representation: Optional[torch.Tensor]


@dataclass
class SITELossComponents:
    total: torch.Tensor
    factual: torch.Tensor
    prediction_error: torch.Tensor
    weight_decay: torch.Tensor
    mid_point: torch.Tensor
    pddm: torch.Tensor




def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")

#计算张量 x 的 L2 范数
def _safe_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x.pow(2).sum(dim=1, keepdim=True), min=eps))

#构建表示层，一个线性、激活、dropout的堆叠
class SITERepresentation(nn.Module):
    """Feature encoder that mirrors the TensorFlow input tower."""

    def __init__(self, config: SITEConfig):
        super().__init__()
        self.config = config
        self.act = _activation(config.activation)
        self.varsel = config.varsel
        #初始化一个可学习的参数向量，用于自动特征选择
        if self.varsel:
            self.feature_gates = nn.Parameter(torch.ones(config.dim_input) / config.dim_input)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if config.batch_norm else None
        for layer_idx in range(config.n_rep_layers):
            in_dim = config.dim_input if layer_idx == 0 else config.dim_rep
            linear = nn.Linear(in_dim, config.dim_rep)
            self.layers.append(linear)
            if self.batch_norms is not None:
                self.batch_norms.append(nn.BatchNorm1d(config.dim_rep))
        self.dropout = nn.Dropout(p=config.dropout_rep)
#前向传播
    def forward(
        self, x: torch.Tensor, pair_samples: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        rep = self._encode_path(x)
        rep_pairs = self._encode_path(pair_samples) if pair_samples is not None else None
        return rep, rep_pairs

    def _encode_path(self, h: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if h is None:
            return None
        if self.varsel:
            h = h * self.feature_gates
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if self.batch_norms is not None:
                h = self.batch_norms[idx](h)
            h = self.act(h)
            h = self.dropout(h)
        if self.config.normalization == "divide":
            h = h / _safe_norm(h)
        return h

    def l2_penalty(self) -> torch.Tensor:
        if not self.config.rep_weight_decay:
            return torch.zeros(1, device=self.layers[0].weight.device if self.layers else "cpu")#这长串代码是为了确保这个“0”是在 GPU 上还是 CPU 上，和模型保持一致
        penalty = torch.zeros(1, device=self.layers[0].weight.device if self.layers else "cpu")
        for layer in self.layers:
            penalty = penalty + layer.weight.pow(2).sum()
        # No penalty on the variable selection gate, matching TensorFlow's behavior.
        return penalty

#输出层
class FeedForwardHead(nn.Module):
    """Helper MLP used by both treatment heads."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, activation: str, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        act = _activation(activation)
        self.activation = act
        self.dropout = nn.Dropout(p=dropout)
        prev_dim = input_dim
        for _ in range(depth):
            layer = nn.Linear(prev_dim, hidden_dim)
            self.layers.append(layer)
            prev_dim = hidden_dim
        self.final = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.dropout(self.activation(layer(h)))
        return self.final(h)

    def l2_penalty(self) -> torch.Tensor:
        penalty = torch.zeros(1, device=self.final.weight.device)
        for layer in self.layers:
            penalty = penalty + layer.weight.pow(2).sum()
        penalty = penalty + self.final.weight.pow(2).sum()
        return penalty

# 输出层，支持TARNet风格的分支或S-learner风格
class TARNetHead(nn.Module):
    """Outcome head that supports either TARNet-style split or S-learner style."""

    def __init__(self, config: SITEConfig):
        super().__init__()
        self.config = config
        if config.split_output:
            self.control_head = FeedForwardHead(config.dim_rep, config.dim_out, config.n_head_layers, config.activation, config.dropout_head)
            self.treated_head = FeedForwardHead(config.dim_rep, config.dim_out, config.n_head_layers, config.activation, config.dropout_head)
        else:
            self.shared_head = FeedForwardHead(config.dim_rep + 1, config.dim_out, config.n_head_layers, config.activation, config.dropout_head)

    def forward(self, rep: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.config.split_output:
            device = rep.device
            y = torch.zeros(rep.size(0), 1, device=device, dtype=rep.dtype)
            mask_control = (t < 0.5).view(-1)#.view(-1): 把形状拉平。通常 t 的形状是 [Batch_Size, 1]（二维矩阵），但作为索引掩码（Mask），我们需要它是一维的 [Batch_Size]。
            mask_treated = ~mask_control
            if mask_control.any():
                y[mask_control] = self.control_head(rep[mask_control])
            if mask_treated.any():
                y[mask_treated] = self.treated_head(rep[mask_treated])
            return y
        rep_with_t = torch.cat([rep, t], dim=1)
        return self.shared_head(rep_with_t)

    def l2_penalty(self) -> torch.Tensor:
        if self.config.split_output:
            return self.control_head.l2_penalty() + self.treated_head.l2_penalty()
        return self.shared_head.l2_penalty()


# --------------------------------------------------------------------------------------
# Full network + losses
# --------------------------------------------------------------------------------------

#总的SITE网络
class SITENetwork(nn.Module):
    """PyTorch module that reproduces the TensorFlow ``site_net`` graph."""

    def __init__(self, config: SITEConfig):
        super().__init__()
        self.config = config
        self.encoder = SITERepresentation(config)
        self.head = TARNetHead(config)
        self.pddm = PDDMNetwork(
            dim_in=config.dim_rep,
            dim_pddm=config.dim_pddm,
            dim_c=config.dim_c,
            dim_s=config.dim_s,
            activation=config.activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        three_pair_samples: Optional[torch.Tensor] = None,
    ) -> SITEForwardOutputs:
        rep, rep_pairs = self.encoder(x, three_pair_samples)
        y = self.head(rep, t)
        return SITEForwardOutputs(y_pred=y, representation=rep, pair_representation=rep_pairs)

    def l2_penalty(self) -> torch.Tensor:
        penalty = self.encoder.l2_penalty() + self.head.l2_penalty()
        penalty = penalty + sum(param.pow(2).sum() for param in self.pddm.parameters())
        return penalty


def _prediction_error(config: SITEConfig, y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if config.loss_type == "l1":
        abs_diff = torch.abs(y_true - y_pred) # $MAE = \frac{1}{N} \sum |y_{true} - y_{pred}|$
        return abs_diff.mean(), abs_diff.mean()
    if config.loss_type == "log":
        y = torch.clamp(torch.sigmoid(y_pred), 0.0025, 0.9975) #Clamp: 截断概率值，防止出现 0 或 1 导致 log(0) 无穷大报错
        res = y_true * torch.log(y) + (1.0 - y_true) * torch.log(1.0 - y) # 负对数似然损失-->计算二元交叉熵
        return (-res).mean(), (-res).mean()
    sq_diff = (y_true - y_pred) ** 2
    return sq_diff.mean(), torch.sqrt(sq_diff.mean())

#用于调节损失函数权重的辅助函数
def _sample_weight(config: SITEConfig, t: torch.Tensor, avg_propensity: float) -> torch.Tensor:
    if not config.reweight_sample:
        return torch.ones_like(t)
    pt = torch.tensor(avg_propensity, dtype=t.dtype, device=t.device)
    w_t = t / (2.0 * pt.clamp_min(1e-6))
    w_c = (1.0 - t) / (2.0 * (1.0 - pt).clamp_min(1e-6))
    return w_t + w_c


def _mid_point_distance(rep_pairs: Optional[torch.Tensor]) -> torch.Tensor:
    if rep_pairs is None or rep_pairs.size(0) < 6:
        return torch.zeros(1, device=rep_pairs.device if rep_pairs is not None else "cpu")
    x_i, x_j, x_k, x_l, x_m, x_n = torch.chunk(rep_pairs, chunks=6, dim=0)
    mid_jk = 0.5 * (x_j + x_k)
    mid_im = 0.5 * (x_i + x_m)
    return (mid_jk - mid_im).pow(2).sum()


def _pddm_penalty(
    config: SITEConfig,
    model: SITENetwork,
    rep_pairs: Optional[torch.Tensor],
    similarity_targets: Optional[Dict[str, float]],
) -> torch.Tensor:
    if rep_pairs is None or similarity_targets is None:
        return torch.zeros(1, device=rep_pairs.device if rep_pairs is not None else "cpu")
    if rep_pairs.size(0) < 6:
        raise ValueError("Expected six pair samples (i, j, k, l, m, n) for PDDM.")
    x_i, x_j, x_k, x_l, x_m, x_n = torch.chunk(rep_pairs, chunks=6, dim=0)
    return compute_pddm_loss(
        model.pddm,
        x_i,
        x_j,
        x_k,
        x_l,
        x_m,
        x_n,
        similarity_targets,
    )

#reg_weights 的作用是提供一组动态的、可配置的缩放系数，用于精细控制各个正则化损失项（Regularization Terms）在总损失中的比重
def compute_site_loss(
    model: SITENetwork,
    outputs: SITEForwardOutputs,
    y_true: torch.Tensor,
    t: torch.Tensor,
    avg_propensity: float,
    similarity_targets: Optional[Dict[str, float]] = None,
    reg_weights: Optional[SITERegularizationWeights] = None,
) -> SITELossComponents:
    """Assemble the full SITE loss mirroring the TensorFlow graph."""

    config = model.config
    reg_weights = reg_weights or SITERegularizationWeights()

    sample_weight = _sample_weight(config, t, avg_propensity)
    factual_loss, pred_error = _prediction_error(config, y_true, outputs.y_pred)
    factual_loss = (sample_weight * factual_loss).mean() if factual_loss.ndim > 0 else factual_loss

    wd = config.p_lambda * reg_weights.weight_decay * model.l2_penalty()
    mid = config.p_mid_point_mini * reg_weights.mid_point * _mid_point_distance(outputs.pair_representation)
    pddm = config.p_pddm * reg_weights.pddm * _pddm_penalty(config, model, outputs.pair_representation, similarity_targets)

    total = factual_loss + wd + mid + pddm
    return SITELossComponents(
        total=total,
        factual=factual_loss,
        prediction_error=pred_error,
        weight_decay=wd,
        mid_point=mid,
        pddm=pddm,
    )


# --------------------------------------------------------------------------------------
# Minimal usage example
# --------------------------------------------------------------------------------------


def example_training_step() -> None:  # pragma: no cover
    batch_size = 128
    dim_input = 51
    torch.manual_seed(0)
    x = torch.randn(batch_size, dim_input)
    t = torch.randint(0, 2, (batch_size, 1)).float()
    y = torch.randn(batch_size, 1)
    three_pair_samples = torch.randn(6, dim_input)

    config = SITEConfig(
        dim_input=dim_input,
        dim_rep=128,
        dim_out=64,
        dim_pddm=64,
        dim_c=64,
        dim_s=1,
        n_rep_layers=2,
        n_head_layers=2,
        split_output=True,
        p_lambda=1e-3,
        p_mid_point_mini=1.0,
        p_pddm=1.0,
    )

    model = SITENetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    outputs = model(x, t, three_pair_samples)
    similarity_targets = {
        "s_kl": 0.9,
        "s_mn": 0.9,
        "s_km": 0.1,
        "s_ik": 0.1,
        "s_jm": 0.1,
    }

    losses = compute_site_loss(
        model,
        outputs,
        y_true=y,
        t=t,
        avg_propensity=float(t.mean().item()),
        similarity_targets=similarity_targets,
    )

    optimizer.zero_grad()
    losses.total.backward()
    optimizer.step()

    print("Loss breakdown", losses)


if __name__ == "__main__":  # pragma: no cover
    example_training_step()
