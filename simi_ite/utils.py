import torch
import numpy as np
from simi_ite.propensity import load_propensity_score
from copy import deepcopy
import math
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from scipy import sparse
import argparse

class StandardScaler:

    # We provide our DIY scaler operator since the treatment column is special
    def __init__(self, protected_indices=None):
        self.mean = 0
        self.std = 0
        self.protected_indices = protected_indices or []

    def _protected_positions(self, dim_size):
        if not self.protected_indices:
            return []
        resolved = []
        for idx in self.protected_indices:
            resolved_idx = idx if idx >= 0 else dim_size + idx
            if 0 <= resolved_idx < dim_size:
                resolved.append(resolved_idx)
        return resolved

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-6
        for pos in self._protected_positions(data.shape[1]):
            self.mean[pos] = 0
            self.std[pos] = 1
        # self.mean[-1] = 0  # Do NOT scale the counterfactual outcome column (it is just used in evaluation)
        # self.std[-1] = 1

        # self.mean = np.zeros_like(self.mean)
        # self.std = np.ones_like(self.mean)

    def transform(self, data):
        data = (data - self.mean) / (self.std)
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def reverse_y(self, yf):
        y = yf * self.std[-2] + self.mean[-2]
        return y

def metric_update(metric: dict(), metric_: dict(), epoch) -> dict():
    """
    Update the metric dict
    :param metric: self.metric in the class Estimator, each value is array
    :param metric_: output of metric() function, each value is float
    :return:
    """
    for key in metric_.keys():
        metric[key] = np.concatenate([metric[key], [metric_[key]]])
    info = "Epoch {:>3}".format(epoch)
    return metric


def metric_export(path, train_metric, eval_metric, test_metric):

    with open(path + '/run.txt', 'w') as f:
        f.write("split,r2_f,rmse_f,mae_f\n")
        f.write("{},{},{},{}\n".format(
            'train',
            train_metric['r2_f'],
            train_metric['rmse_f'],
            train_metric['mae_f']
        ))
        f.write("{},{},{},{}\n".format(
            'eval',
            eval_metric['r2_f'],
            eval_metric['rmse_f'],
            eval_metric['mae_f']
        ))

        f.write("{},{},{},{}\n".format(
            'test',
            test_metric['r2_f'],
            test_metric['rmse_f'],
            test_metric['mae_f']
        ))


def metrics(
        pred_0: np.ndarray,
        # pred_1: np.ndarray,
        yf: np.ndarray,
        # ycf: np.ndarray,
        # t: np.ndarray,
        ) -> dict:

    assert len(pred_0.shape) == 1
    # assert len(pred_1.shape) == 1
    assert len(yf.shape) == 1 #and len(ycf.shape) == 1
    # assert len(t.shape) == 1
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


    # length = len(t)
    # y0 = yf * (1-t) + ycf * t
    # y1 = yf * t + ycf * (1-t)

    # Section: factual fitting
    # yf_pred = pred_1 * t + pred_0 * (1-t)
    r2_f = r2_score(yf,pred_0)
    rmse_f = np.sqrt(mean_squared_error(yf, pred_0))
    mae_f = mean_absolute_error(yf, pred_0)

    # Section: counterfactual fitting
    # ycf_pred = pred_0 * t + pred_1 * (1-t)
    # r2_cf = r2_score(ycf, ycf_pred)
    # rmse_cf = np.sqrt(mean_squared_error(ycf, ycf_pred))

    # Section: ITE estimation
    # _pred_0 = deepcopy(pred_0)
    # _pred_1 = deepcopy(pred_1)
    # if mode == "in-sample":
    #     _pred_0[t == 0] = y0[t == 0]
    #     _pred_1[t == 1] = y1[t == 1]
    # effect_pred = _pred_1 - _pred_0
    # effect = y1 - y0

    # # Negative effect
    # effect_pred = effect_pred
    # effect = effect

    # pehe = np.sqrt(np.mean((effect - effect_pred) ** 2))
    # ate = np.mean(effect)
    # ate_pred = np.mean(effect_pred)
    # att = np.mean(effect[t == 1])
    # att_pred = np.mean(effect_pred[t == 1])
    # mae_ate = np.abs(ate - ate_pred)
    # mae_att = np.abs(att - att_pred)
    # auuc = auuc_score(yf=yf, t=t, effect_pred=effect_pred)


    return {
        # "mae_ate": round(mae_ate, 5),
        # "mae_att": round(mae_att, 5),
        # "pehe": round(pehe, 5),
        "r2_f": round(r2_f, 5),
        "rmse_f": round(rmse_f, 5),
        "mae_f": round(mae_f, 5),
        # "r2_cf": round(r2_cf, 5),
        # "rmse_cf": round(rmse_cf, 5),
        # "auuc": round(auuc[0], 5),
        # "rauuc": round(auuc[1], 5)
    }

# def load_sparse(fname):
#     """ Load sparse data set """
#     E = np.loadtxt(open(fname, "rb"), delimiter=",")
#     H = E[0, :]
#     n = int(H[0])
#     d = int(H[1])
#     E = E[1:, :]
#     S = sparse.coo_matrix((E[:, 2], (E[:, 0]-1, E[:, 1]-1)), shape=(n, d))
#     S = S.todense() 

#     return S

# 计算相似度得分
def similarity_score(s_i, s_j):
    # if mode == 'sigmoid':
    #     _mid = (s_i + s_j)/float(2)
    #     _dis = abs(s_j - s_i)/float(2)
    #     score = 2*sigmoid(abs(_mid-0.5)) - 3*sigmoid(_dis)+1
    # if mode == 'linear':
    _mid = (s_i + s_j) / float(2)
    _dis = abs(s_j - s_i) / float(2)
    score = (1.5 * abs(_mid - 0.5) - 2 * _dis + 1)/float(2)
    return score

# 用于 PDDM/SITe 损失
def row_wise_dist(x):#高效计算矩阵 $x$ 内部样本两两之间的 平方欧几里得距离。
    r = torch.sum(x * x, dim=1)
    # turn r into column vector
    r = torch.reshape(r, [-1, 1])
    D = r - 2 * torch.matmul(x, x.t()) + r.t()
    return D
#构建 “倾向性相似度”矩阵，作为训练的 Ground Truth（标准答案）。
def get_simi_ground(x, propensity_model):
    x_propensity_score = propensity_model.predict_proba(x)
    n_train = x.shape[0]
    s_similarity_matrix = np.ones([n_train, n_train])
    if x_propensity_score.ndim == 1:
        s = x_propensity_score.reshape(-1, 1)
    else:
        s = x_propensity_score
    mid_matrix = (s + s.T) / 2.0
    dis_matrix = np.abs(s - s.T) / 2.0
    s_similarity_matrix = (1.5 * np.abs(mid_matrix - 0.5) - 2.0 * dis_matrix + 1.0) / 2.0
    # for i in range(n_train):
    #     for j in range(n_train):
    #         s_similarity_matrix[i, j] = similarity_score(x_propensity_score[i], x_propensity_score[j])
    return x_propensity_score, s_similarity_matrix

# def find_nearest_point(x, p):#在数组 $x$ 中，找到与数值 $p$ 最接近的另一个点的索引
#     diff = np.abs(x-p)
#     diff_1 = diff[diff>0]
#     min_val = np.min(diff_1)
#     I_diff = np.where(diff == min_val)[0]
#     I_diff = I_diff[0]
#     if I_diff.size > 1:
#         I_diff = I_diff[0]
#     return I_diff
def find_nearest_point(x, p, exclude_index=None):
    """Return the index of the element in ``x`` closest to ``p`` while optionally excluding one index."""
    diff = np.abs(x - p)
    if exclude_index is not None and 0 <= exclude_index < diff.shape[0]:
        diff = diff.copy()
        diff[exclude_index] = np.inf
    candidate = np.argmin(diff)
    if np.isinf(diff[candidate]):
        raise ValueError("Unable to find a distinct neighbour for propensity score selection.")
    return candidate

def find_middle_pair(x, y):
    # min_val = np.abs(x[0]-0.5) + np.abs(y[0]-0.5)
    # index_1 = 0
    # index_2 = 0
    # for i in range(x.shape[0]):
    #     for j in range(y.shape[0]):
    #         value = np.abs(x[i]-0.5) + np.abs(y[j]-0.5)
    #         if value < min_val:
    #             min_val = value
    #             index_1 = i
    #             index_2 = j
    index_1 = np.argmin(np.abs(x - 0.5))
    index_2 = np.argmin(np.abs(y - 0.5))
    # x_dev = np.abs(x - 0.5)[:, None]    # shape (n,1)
    # y_dev = np.abs(y - 0.5)[None, :]    # shape (1,m)
    # total_dev = x_dev + y_dev           # 广播得到 n×m
    # idx = np.argmin(total_dev)
    # index_1, index_2 = np.unravel_index(idx, total_dev.shape)
    return index_1, index_2

# def find_three_pairs(x, t, x_propensity_score):
#     try:
#         x_return = np.ones([6, x.shape[1]])
#         I_x_return = np.zeros(6, dtype=int)
#         # x_propensity_score = load_propensity_score(propensity_dir, x)
#         I_t = np.where(t > 0)[0]
#         I_c = np.where(t < 1)[0]

#         prop_t = x_propensity_score[I_t]
#         prop_c = x_propensity_score[I_c]
        
#         x_t = x[I_t]
#         x_c = x[I_c]
        
#         # find x_i, x_j
#         index_t, index_c = find_middle_pair(prop_t, prop_c)
#         # find x_k, x_l
#         index_k = np.argmax(np.abs(prop_c - prop_t[index_t]))
#         index_l = find_nearest_point(prop_c, prop_c[index_k])

#         # find x_n, x_m
#         index_m = np.argmax(np.abs(prop_t - prop_c[index_c]))
#         index_n = find_nearest_point(prop_t, prop_t[index_m,])
        
#         x_return[0, :] = x_t[index_t, :]
#         x_return[1, :] = x_c[index_c, :]
#         x_return[2, :] = x_c[index_k, :]
#         x_return[3, :] = x_c[index_l, :]
#         x_return[4, :] = x_t[index_m, :]
#         x_return[5, :] = x_t[index_n, :]
#         I_x_return[0] = int(I_t[index_t])
#         I_x_return[1] = int(I_c[index_c])
#         I_x_return[2] = int(I_c[index_k])
#         I_x_return[3] = int(I_c[index_l])
#         I_x_return[4] = int(I_t[index_m])
#         I_x_return[5] = int(I_t[index_n])
#     except:
#         x_return = x[0:6, :]
#         I_x_return = np.array([0, 1, 2, 3, 4, 5])
#         print('some error happens here!')

#     return x_return, I_x_return
def find_three_pairs(x, t, x_propensity_score):
    x = np.asarray(x)
    t = np.asarray(t).reshape(-1)
    prop_scores = np.asarray(x_propensity_score)
    if prop_scores.ndim > 1:
        prop_scores = prop_scores[:, -1]

    treated_idx = np.where(t > 0)[0]
    control_idx = np.where(t < 1)[0]
    if treated_idx.size < 2 or control_idx.size < 2:
        raise ValueError("Need at least two treated and two control samples per batch to build triplets.")

    prop_t = prop_scores[treated_idx]
    prop_c = prop_scores[control_idx]
    x_t = x[treated_idx]
    x_c = x[control_idx]

    index_t, index_c = find_middle_pair(prop_t, prop_c)
    index_k = np.argmax(np.abs(prop_c - prop_t[index_t]))
    index_l = find_nearest_point(prop_c, prop_c[index_k], exclude_index=index_k)
    index_m = np.argmax(np.abs(prop_t - prop_c[index_c]))
    index_n = find_nearest_point(prop_t, prop_t[index_m], exclude_index=index_m)

    idx_t = int(treated_idx[index_t])
    idx_c = int(control_idx[index_c])
    idx_k = int(control_idx[index_k])
    idx_l = int(control_idx[index_l])
    idx_m = int(treated_idx[index_m])
    idx_n = int(treated_idx[index_n])

    x_return = np.zeros((6, x.shape[1]), dtype=x.dtype)
    I_x_return = np.zeros(6, dtype=int)
    x_return[0, :] = x_t[index_t]
    x_return[1, :] = x_c[index_c]
    x_return[2, :] = x_c[index_k]
    x_return[3, :] = x_c[index_l]
    x_return[4, :] = x_t[index_m]
    x_return[5, :] = x_t[index_n]
    I_x_return[:] = [idx_t, idx_c, idx_k, idx_l, idx_m, idx_n]

    return x_return, I_x_return

def get_three_pair_simi(similarity_ground, three_pairs_index):
    simi = np.ones([5, 1])
    '''
    S(k, l), S(m, n), S(k, l), S(i, k), S(j, m)
    '''
    simi[0, 0] = similarity_ground[three_pairs_index[2], three_pairs_index[3]]
    simi[1, 0] = similarity_ground[three_pairs_index[4], three_pairs_index[5]]
    simi[2, 0] = similarity_ground[three_pairs_index[2], three_pairs_index[4]]
    simi[3, 0] = similarity_ground[three_pairs_index[0], three_pairs_index[2]]
    simi[4, 0] = similarity_ground[three_pairs_index[1], three_pairs_index[4]]
    return simi

def _sample_weight(hparams, t: torch.Tensor, avg_propensity: float) -> torch.Tensor:
    if not hparams.get('reweight_sample', False):
        return torch.ones_like(t)
    pt = torch.tensor(avg_propensity, dtype=t.dtype, device=t.device)
    w_t = t / (2.0 * pt.clamp_min(1e-6))
    w_c = (1.0 - t) / (2.0 * (1.0 - pt).clamp_min(1e-6))
    return w_t + w_c


def _load_propensity_model(self, input_dim,hparams):
    directory = Path(hparams.get('propensity_dir', 'simi_ite/tmp/propensity_model'))
    if not directory.exists():
        raise FileNotFoundError(f"Propensity model directory '{directory}' does not exist.")
    meta_file = directory / 'meta.txt'
    mode = meta_file.read_text().strip() if meta_file.exists() else 'logistic'
    return load_propensity_score(directory, input_dim=input_dim, mode=mode)


if __name__ == "__main__":
    pass