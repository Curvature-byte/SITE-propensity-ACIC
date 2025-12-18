import torch
import torch.nn as nn
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from simi_ite.pddm_net import PDDMNetwork, compute_pddm_loss
# class SITEConfig:
#     """Hyper-parameters that mirror the TensorFlow ``FLAGS`` + ``dims`` arrays."""

#     dim_pddm_in: int = 32
#     dim_pddm_hide: int = 16
#     dim_pddm_c: int = 32
#     varsel: bool = False
#     batch_norm: bool = False
#     normalization: str = "none"  # {"none", "divide"}
#     loss_type: str = "l2"  # {"l2", "l1", "log"}
#     reweight_sample: bool = False
#     rep_weight_decay: bool = True
#     dropout_rep: float = 0.1  # dropout probability for representation backbone
#     dropout_head: float = 0.1  # dropout probability for outcome head
#     p_lambda: float = 0.0
#     p_mid_point_mini: float = 0.0
#     p_pddm: float = 0.0


# @dataclass
# class SITERegularizationWeights:
#     """Runtime scalars that multiply the base ``p_*`` coefficients."""

#     weight_decay: float = 1.0
#     mid_point: float = 1.0
#     pddm: float = 1.0


# @dataclass
# class SITEForwardOutputs:
#     """Convenience container so the loss function receives consistent tensors."""

#     y_pred: torch.Tensor
#     representation: torch.Tensor
#     pair_representation: Optional[torch.Tensor]


# @dataclass
# class SITELossComponents:
#     total: torch.Tensor
#     factual: torch.Tensor
#     prediction_error: torch.Tensor
#     weight_decay: torch.Tensor
#     mid_point: torch.Tensor
#     pddm: torch.Tensor
# #计算张量 x 的 L2 范数
# def _safe_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#     return torch.sqrt(torch.clamp(x.pow(2).sum(dim=1, keepdim=True), min=eps))


class LinearLearner(nn.Module):  # for test

    def __init__(self, input_dim=50, params={}):

        super(LinearLearner, self).__init__()
        self.output = nn.Linear(input_dim+1, 1)

    def forward(self, x):

        output = self.output(x)
        return output



class Representation(nn.Module):
    """
    Representation network for SITE
    """
    def __init__(self, input_dim, hparams):

        super(Representation, self).__init__()
        self.hparams = hparams
        self.batch_norm = hparams.get('batch_norm', False)
        out_backbone = hparams.get('dim_backbone', '32,16').split(',')
        in_backbone = [input_dim] + list(map(int, out_backbone))
        self.backbone = torch.nn.Sequential()
        if hparams.get('varsel', False):
            self.feature_gates = nn.Parameter(torch.ones(input_dim) / input_dim)
        for i in range(1, len(in_backbone)):
            self.backbone.add_module(f"backbone_dense{i}", torch.nn.Linear(in_backbone[i-1], in_backbone[i]))
            self.backbone.add_module(f"backbone_relu{i}", torch.nn.ELU())
            if self.batch_norm:
                self.backbone.add_module(f"backbone_bn{i}", torch.nn.BatchNorm1d(num_features=in_backbone[i]))
        self.backbone.add_module(f"backbone_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

    def forward(self, x):
        if self.hparams.get('varsel', False):
            x = x * self.feature_gates
        
        rep = self.backbone(x)

        return rep

class output_head(nn.Module):
    """
    Outcome head for SITE
    """
    def __init__(self, hparams):

        super(output_head, self).__init__()
        self.hparams = hparams
        self.batch_norm = hparams.get('batch_norm', False)
        out_backbone = hparams.get('dim_backbone', '32,16').split(',')
        out_task = hparams.get('dim_task', '16').split(',')
        self.treat_embed = hparams.get('treat_embed', True)
        out_backbone =list(map(int, out_backbone))
        in_task = [out_backbone[-1]] + list(map(int, out_task))
        if self.treat_embed is True: # 拼接treatment带来的
            in_task[0] += 2

        self.tower = torch.nn.Sequential()
        for i in range(1, len(in_task)):
            self.tower.add_module(f"tower_dense{i}", torch.nn.Linear(in_task[i-1], in_task[i]))
            self.tower.add_module(f"tower_relu{i}", torch.nn.ELU())
            if self.batch_norm:
                self.tower.add_module(f"tower_bn{i}", torch.nn.BatchNorm1d(num_features=in_task[i]))
            self.tower.add_module(f"tower_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.output = torch.nn.Sequential()
        self.output.add_module("output_dense", torch.nn.Linear(in_task[-1], 1))

        self.embedding = nn.Embedding(2, 2)

    def forward(self, rep: torch.Tensor,x: torch.Tensor):
        t = x[:, -1]
        if self.treat_embed is True:
            t_embed = self.embedding(t.int())
            rep_t = torch.cat([rep, t_embed], dim=-1)
        else:
            rep_t = rep
        outcome_f = self.output(self.tower(rep_t))
        return outcome_f
# 总的网络结构
class TARNetHead(nn.Module):
    def __init__(self, input_dim, hparams):
        super(TARNetHead, self).__init__()
        self.hparams = hparams
        out_backbone = hparams.get('dim_backbone', '32,16').split(',')
        out_task = hparams.get('dim_task', '16').split(',')
        self.treat_embed = hparams.get('treat_embed', True)
        in_backbone = [input_dim] + list(map(int, out_backbone))
        in_task = [in_backbone[-1]] + list(map(int, out_task))
        if self.treat_embed is True: # 拼接treatment带来的
            in_task[0] += 2
        self.rep=Representation(input_dim, hparams)
        self.head0 = output_head(hparams)
        self.head1 =  deepcopy(self.head0)
    


    def forward(self, x: torch.Tensor):
        covariates = x[:, :-1]
        t = x[:, -1]
        rep = self.rep(covariates) 
        self.rep_1 = rep[t == 1]
        self.rep_0 = rep[t == 0]

        self.out_1 = self.head1(rep, x)
        self.out_0 = self.head0(rep, x)
        
        t = t.reshape(-1, 1)
        output_f = t * self.out_1 + (1 - t) * self.out_0

        return output_f



class YLearner(nn.Module):
    """
    TARNet which combines T-learner and S-learner.
    """
    def __init__(self, input_dim, hparams):

        super(YLearner, self).__init__()

        out_backbone = hparams.get('dim_backbone', '32,16').split(',')
        out_task = hparams.get('dim_task', '16').split(',')
        self.treat_embed = hparams.get('treat_embed', True)
        in_backbone = [input_dim] + list(map(int, out_backbone))
        self.batch_norm = hparams.get('batch_norm', False)
        self.backbone = torch.nn.Sequential()
        for i in range(1, len(in_backbone)):
            self.backbone.add_module(f"backbone_dense{i}", torch.nn.Linear(in_backbone[i-1], in_backbone[i]))
            self.backbone.add_module(f"backbone_relu{i}", torch.nn.ELU())
            if self.batch_norm:
                self.backbone.add_module(f"backbone_bn{i}", torch.nn.BatchNorm1d(num_features=in_backbone[i]))
            self.backbone.add_module(f"backbone_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        in_task = [in_backbone[-1]] + list(map(int, out_task))
        if self.treat_embed is True: # 拼接treatment带来的
            in_task[0] += 2

        self.tower = torch.nn.Sequential()
        for i in range(1, len(in_task)):
            self.tower.add_module(f"tower_dense{i}", torch.nn.Linear(in_task[i-1], in_task[i]))
            self.tower.add_module(f"tower_relu{i}", torch.nn.ELU())
            if self.batch_norm:
                self.tower.add_module(f"tower_bn{i}", torch.nn.BatchNorm1d(num_features=in_task[i]))
            self.tower.add_module(f"tower_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.output = torch.nn.Sequential()
        self.output.add_module("output_dense", torch.nn.Linear(in_task[-1], 1))

        self.tower_0 = deepcopy(self.tower)
        self.output_0 = deepcopy(self.output)

        self.rep_1, self.rep_0 = None, None
        self.out_1, self.out_0 = None, None
        self.embedding = nn.Embedding(2, 2)

    def forward(self, x: torch.Tensor,pair_samples: Optional[torch.Tensor] = None)-> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        rep_pairs = self.backbone(pair_samples) if pair_samples is not None else None
        covariates = x[:, :-1]
        t = x[:, -1]
        rep = self.backbone(covariates)
        if self.treat_embed is True:
            t_embed = self.embedding(t.int())
            rep_t = torch.cat([rep, t_embed], dim=-1)
        else:
            rep_t = rep

        self.rep_1 = rep[t == 1]
        self.rep_0 = rep[t == 0]

        self.out_1 = self.output(self.tower(rep_t))
        self.out_0 = self.output_0(self.tower_0(rep_t))

        t = t.reshape(-1, 1)
        output_f = t * self.out_1 + (1 - t) * self.out_0
        return output_f , rep_pairs
    # def l2_penalty(self) -> torch.Tensor:
    #     if not self.config.rep_weight_decay:
    #         return torch.zeros(1, device=self.backbone[0].weight.device if self.backbone else "cpu")#这长串代码是为了确保这个“0”是在 GPU 上还是 CPU 上，和模型保持一致
    #     penalty = torch.zeros(1, device=self.backbone[0].weight.device if self.backbone else "cpu")
    #     for layer in self.backbone:
    #         if isinstance(layer, nn.Linear):
    #             penalty = penalty + layer.weight.pow(2).sum()
    #     # No penalty on the variable selection gate, matching TensorFlow's behavior.
    #     penalty += self.embedding.weight.pow(2).sum()
    #     for layer in self.tower:
    #         if isinstance(layer, nn.Linear):
    #             penalty = penalty + layer.weight.pow(2).sum()
    #     for layer in self.tower_0:
    #         if isinstance(layer, nn.Linear):
    #             penalty = penalty + layer.weight.pow(2).sum()
    #     return penalty
#总的SITE网络结构
# class SITENetwork(nn.Module):
#     """PyTorch module that reproduces the TensorFlow ``site_net`` graph."""

#     def __init__(self):
#         super().__init__()

#         self.encoder = YLearner(input_dim=30, hparams={})
#         self.head = SLearner(input_dim=30, hparams={})
#         self.pddm = PDDMNetwork(
#             dim_in=config.dim_pddm,
#             dim_pddm=config.dim_pddm_hide,
#             dim_c=config.dim_pddm_c,
#             hparams={}
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#         t: torch.Tensor,
#         three_pair_samples: Optional[torch.Tensor] = None,
#     ):
#         y, rep_pairs = self.encoder(x, three_pair_samples)
#         return y,rep_pairs

#     def l2_penalty(self) -> torch.Tensor:
#         penalty = self.encoder.l2_penalty() 
#         penalty = penalty + sum(param.pow(2).sum() for param in self.pddm.parameters())
#         return penalty
    


# def _prediction_error(config: SITEConfig, y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     if config.loss_type == "l1":
#         abs_diff = torch.abs(y_true - y_pred) # $MAE = \frac{1}{N} \sum |y_{true} - y_{pred}|$
#         return abs_diff.mean(), abs_diff.mean()
#     if config.loss_type == "log":
#         y = torch.clamp(torch.sigmoid(y_pred), 0.0025, 0.9975) #Clamp: 截断概率值，防止出现 0 或 1 导致 log(0) 无穷大报错
#         res = y_true * torch.log(y) + (1.0 - y_true) * torch.log(1.0 - y) # 负对数似然损失-->计算二元交叉熵
#         return (-res).mean(), (-res).mean()
#     sq_diff = (y_true - y_pred) ** 2
#     return sq_diff.mean(), torch.sqrt(sq_diff.mean())

# #用于调节损失函数权重的辅助函数
# def _sample_weight(config: SITEConfig, t: torch.Tensor, avg_propensity: float) -> torch.Tensor:
#     if not config.reweight_sample:
#         return torch.ones_like(t)
#     pt = torch.tensor(avg_propensity, dtype=t.dtype, device=t.device)
#     w_t = t / (2.0 * pt.clamp_min(1e-6))
#     w_c = (1.0 - t) / (2.0 * (1.0 - pt).clamp_min(1e-6))
#     return w_t + w_c

# def _pddm_penalty(
#     pddm_net: PDDMNetwork,
#     rep_pairs: np.ndarray,
#     similarity_targets: Dict[str, float]
# ) -> torch.Tensor:
#     if rep_pairs is None or similarity_targets is None:
#         return torch.zeros(1, device=rep_pairs.device if rep_pairs is not None else "cpu")
#     if rep_pairs.size(0) < 6:
#         raise ValueError("Expected six pair samples (i, j, k, l, m, n) for PDDM.")
#     x_i, x_j, x_k, x_l, x_m, x_n = torch.chunk(rep_pairs, chunks=6, dim=0)
#     three_pairs = torch.cat([x_i, x_j, x_k, x_l, x_m, x_n], dim=0)
#     return compute_pddm_loss(
#         pddm_net,
#         three_pairs,
#         similarity_targets,
#     )
# #reg_weights 的作用是提供一组动态的、可配置的缩放系数，用于精细控制各个正则化损失项（Regularization Terms）在总损失中的比重
# def compute_site_loss(
#     model: TARNetHead,
#     outputs: SITEForwardOutputs,
#     y_true: torch.Tensor,
#     t: torch.Tensor,
#     avg_propensity: float,
#     similarity_targets: Optional[Dict[str, float]] = None,
#     reg_weights: Optional[SITERegularizationWeights] = None,
# ) -> SITELossComponents:
#     """Assemble the full SITE loss mirroring the TensorFlow graph."""

#     config = model.config
#     reg_weights = reg_weights or SITERegularizationWeights()

#     sample_weight = _sample_weight(config, t, avg_propensity)
#     factual_loss, pred_error = _prediction_error(config, y_true, outputs.y_pred)
#     factual_loss = (sample_weight * factual_loss).mean() if factual_loss.ndim > 0 else factual_loss

#     wd = config.p_lambda * reg_weights.weight_decay * model.l2_penalty()
#     mid = config.p_mid_point_mini * reg_weights.mid_point * _mid_point_distance(outputs.pair_representation)
#     pddm = config.p_pddm * reg_weights.pddm * _pddm_penalty(config, model, outputs.pair_representation, similarity_targets)

#     total = factual_loss + wd + mid + pddm
#     return SITELossComponents(
#         total=total,
#         factual=factual_loss,
#         prediction_error=pred_error,
#         weight_decay=wd,
#         mid_point=mid,
#         pddm=pddm,
#     )
