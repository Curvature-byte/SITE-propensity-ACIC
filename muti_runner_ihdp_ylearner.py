# coding=utf-8
import argparse
import numpy as np
from copy import deepcopy
from pathlib import Path
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from simi_ite.data_processor_test import MyDataset
from models import *
from simi_ite.utils import find_three_pairs, get_simi_ground,  get_three_pair_simi, similarity_score
from simi_ite.utils_test import  metric_update, metrics
from simi_ite.pddm_net import compute_pddm_loss, PDDMNetwork
from simi_ite.propensity import load_propensity_score
import ot


def cal_wass(rep_0, rep_1, out_0, out_1, t, yf, device, hparams):
    yf = yf.unsqueeze(-1)
    dist = hparams['ot_scale'] * ot.dist(rep_0, rep_1)

    if hparams['gamma'] > 0:

        pred_0_cf = out_1[t == 0]  # predicted outcome for samples in control group given t == 1对照组中样本的预测反事实结果（T=0）
        pred_1_cf = out_0[t == 1]  # predicted outcome for samples in treated group given t == 0处理组中样本的预测反事实结果（T=1）
        yf_1 = yf[t == 1]
        yf_0 = yf[t == 0]

        dist_10 = ot.dist(pred_0_cf, yf_1)
        dist_01 = ot.dist(yf_0, pred_1_cf)

        dist += hparams['gamma'] * (dist_01 + dist_10)

    if hparams['ot'] == 'ot':

        gamma = ot.sinkhorn(
            torch.ones(len(rep_0), device=device) / len(rep_0),
            torch.ones(len(rep_1), device=device) / len(rep_1),
            dist.detach(),
            reg=hparams.get('epsilon'),
            stopThr=1e-4)

    elif hparams['ot'] == 'uot':

        gamma = ot.unbalanced.sinkhorn_unbalanced(
            torch.ones(len(rep_0), device=device) / len(rep_0),
            torch.ones(len(rep_1), device=device) / len(rep_1),
            dist.detach(),
            reg=hparams.get('epsilon'),
            stopThr=1e-6,
            reg_m=hparams.get('kappa'))
    else:
        print("ERROR: The hparams.ot is not correctly defined")

    loss_wass = torch.sum(gamma * dist)
    return loss_wass
class BaseEstimator:

    def __init__(self, board=None, hparams={}):
        data_name = hparams.get('data')
        print("Current data:", data_name)
        # train_set = MyDataset(f"Datasets/{data_name}/train.csv")
        # # shared_scaler = train_set.scaler
        # traineval_set = MyDataset(f"Datasets/{data_name}/traineval.csv")
        # eval_set = MyDataset(f"Datasets/{data_name}/eval.csv")
        # test_set = MyDataset(f"Datasets/{data_name}/test.csv")
        train_set = MyDataset(f"Datasets/{data_name}/train_{hparams['seed']}.csv")
        traineval_set = MyDataset(f"Datasets/{data_name}/traineval_{hparams['seed']}.csv")
        eval_set = MyDataset(f"Datasets/{data_name}/eval_{hparams['seed']}.csv")
        test_set = MyDataset(f"Datasets/{data_name}/test_{hparams['seed']}.csv")
        self.device = torch.device(hparams.get('device'))
        if hparams['treat_weight'] == 0:
            self.train_loader = DataLoader(train_set, batch_size=hparams.get('batchSize'), drop_last=True,shuffle=True)
        else:
            self.train_loader = DataLoader(train_set, batch_size=hparams.get('batchSize'), sampler=train_set.get_sampler(hparams['treat_weight']), drop_last=True)
        self.traineval_data = DataLoader(traineval_set, batch_size=256)  # for test in-sample metric
        self.eval_data = DataLoader(eval_set, batch_size=256,shuffle=True)
        self.test_data = DataLoader(test_set, batch_size=256,shuffle=True)


        self.eval_metric = {
             "mae_ate": np.array([]),
             "mae_att": np.array([]),
             "pehe": np.array([]),
             "r2_f": np.array([]),
             "rmse_f": np.array([]),
             "r2_cf": np.array([]),
             "rmse_cf": np.array([]),
             "auuc": np.array([]),
             "rauuc": np.array([])}
        # self.eval_metric = deepcopy(self.train_metric)
        # self.test_metric = deepcopy(self.train_metric)

        self.train_best_metric = {
             "mae_ate": None,
             "mae_att": None,
             "pehe": None,
             "r2_f": None,
             "rmse_f": None,
             "r2_cf": None,
             "rmse_cf": None,
             "auuc": None,
             "rauuc": None,}
        self.eval_best_metric = deepcopy(self.train_best_metric)
        self.eval_best_metric['r2_f'] = -10  
        self.eval_best_metric["pehe"] = 100
        self.eval_best_metric['auuc'] = 0
        self.test_best_metric = deepcopy(self.train_best_metric)
        # loss_metric = {'loss': np.array([]), 'loss_f': np.array([]), 'loss_pddm': np.array([]), 'loss_c': np.array([])}

        self.epochs = hparams.get('epoch', 200)
        # self._pddm_warning_emitted = False
        self.model =  YLearner(train_set.x_dim, hparams).to(self.device)
        # self.rep_model = self.model.rep
        # self.propensity_model = load_propensity_score(model_dir = './simi_ite/IHDP/propensity_model' ,input_dim = train_set.x_dim ,mode="mlp")
        # self.pddm_model = PDDMNetwork(
        #     dim_in=hparams['dim_pddm_in'],
        #     dim_pddm=hparams['dim_pddm_hide'],
        #     dim_c=hparams['dim_pddm_c'],
        # ).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams.get('lr', 1e-3),
            weight_decay=hparams.get('l2_reg', 1e-4)
        )
        # optimizer = torch.optim.Adam(model.parameters(), lr=hparams.get('lr', 1e-3), weight_decay=hparams.get('l2_reg', 1e-4))
        self.hparams = hparams
        self.epoch = 0
        # board = board

    def fit( self):

        iter_num = 0
        stop_epoch = 0 # record how many iterations the eval metrics do not improve
        for epoch in tqdm(range(1, self.epochs)):

            self.epoch = epoch
            self.model.train()
            # self.pddm_model.train()
            for data in self.train_loader:  # train_loader
                self.model.zero_grad()
                data = data.to(self.device)
                _x, _xt, _t, _yf, _= data[:,:-3], data[:, :-2], data[:, -3], data[:, -2],  data[:, -1]
                # _x, _xt, _t, _yf = data[:, 1:-3], data[:, 1:-2], data[:, -3], data[:, -1]
                _pred_f = self.model(_xt).squeeze(-1)
                _loss_fit = self.criterion(_pred_f, _yf)
                _loss_wass = 0
                wass_indicator = (self.hparams['ot'] != 'none' and epoch > 20 and len(_t.unique()) > 1)
                if wass_indicator: # Avoid samples coming from same group
                    _loss_wass = cal_wass(rep_0=self.model.rep_0,
                                         rep_1=self.model.rep_1,
                                          out_0=self.model.out_0,
                                          out_1=self.model.out_1,
                                          t=_t,
                                          yf=_yf,
                                          device=self.device,
                                          hparams=self.hparams)

                # treated = int((_t > 0.5).sum().item())
                # control = int((_t < 0.5).sum().item())
                # pddm_indicator = (self.epoch >40 and treated >= 2 and control >= 2)

                # loss_pddm_tensor = torch.tensor(0.0, device=self.device)
                # # loss_pddm_value = 0.0

                # if pddm_indicator:
                #     x_np = _x.detach().cpu().numpy()
                #     t_np = _t.detach().cpu().numpy()
                #     _x_propensity_scores, _x_simi_ground = get_simi_ground(x_np, propensity_model=self.propensity_model)
                #     try:
                #         three_pairs, three_paris_index = find_three_pairs(x_np, t_np, _x_propensity_scores)
                #     except ValueError as exc:
                #         if not self._pddm_warning_emitted:
                #             print(f"[SITE] Skip PDDM for this batch: {exc}")
                #             self._pddm_warning_emitted = True
                #     else:
                #         three_pairs = torch.tensor(three_pairs, dtype=torch.float32, device=self.device)
                #         three_pairs_rep = self.rep_model(three_pairs)
                #         loss_pddm_tensor = compute_pddm_loss(
                #             self.pddm_model,
                #             three_pairs_rep,
                #             _x_simi_ground,
                #             three_paris_index
                #         ).squeeze()
                #         # loss_pddm_value = loss_pddm_tensor.item()

                _loss = _loss_fit + self.hparams['lambda'] * _loss_wass  #+ self.hparams['lambda_pddm'] * loss_pddm_tensor 
                _loss.backward()
                self.optimizer.step()
                iter_num += 1

           
            if self.epoch % 1 == 0:
                _eval_metric = self.evaluation(data='eval')

                if abs(_eval_metric['auuc']) > abs(self.eval_best_metric['auuc']):
                # if abs(_eval_metric['pehe']) < abs(self.eval_best_metric['pehe']):
                    self.eval_best_metric = _eval_metric
                    self.train_best_metric = self.evaluation(data='train')
                    self.test_best_metric = self.evaluation(data='test')
                    stop_epoch = 0
                    print(self.eval_best_metric)
                else:
                    stop_epoch += 1
            if stop_epoch >= self.hparams['stop_epoch'] and self.epoch > 200:
                                             
                print(f'Early stop at epoch {self.epoch}')
                break

            self.epoch += 1
        return self.train_best_metric, self.eval_best_metric


    def predict(self, dataloader):
        """

        :param dataloader
        :return: np.array, shape: (#sample)
        """
        self.model.eval()
        pred_0 = torch.tensor([], device=self.device)
        yf = deepcopy(pred_0)
        pred_1, yf, ycf, t = deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0),
 
        for data in dataloader:
            data = data.to(self.device)
            _x, _t, _yf, _ycf = data[:, :-3], data[:, [-3]], data[:, -2], data[:, -1]
            _x_0 = torch.cat([_x, torch.zeros_like(_t, device=self.device)], dim=-1)
            _x_1 = torch.cat([_x, torch.ones_like(_t, device=self.device)], dim=-1)
            _pred_0 = self.model(_x_0).reshape([-1])
            _pred_1 = self.model(_x_1).reshape([-1])
            pred_0 = torch.cat([pred_0, _pred_0], axis=-1)
            pred_1 = torch.cat([pred_1, _pred_1], axis=-1)
            yf = torch.cat([yf, _yf], axis=-1)
            ycf = torch.cat([ycf, _ycf], axis=-1)
            t = torch.cat([t, _t.reshape([-1])], axis=-1)

        pred_0 = pred_0.detach().cpu().numpy()
        pred_1 = pred_1.detach().cpu().numpy()
        yf = yf.cpu().numpy()
        # pred_0 = scaler.reverse_y(pred_0)
        # yf = scaler.reverse_y(yf)
        ycf = ycf.cpu().numpy()
        t = t.detach().cpu().numpy()
        return pred_0, pred_1, yf, ycf, t

    def evaluation(self, data: str) -> dict():

        dataloader = {
            'train': self.traineval_data,
            'eval': self.eval_data,
            'test': self.test_data}[data]
        # pred_0, yf, pred_1, yf, ycf, t = predict(dataloader)
        pred_0, pred_1, yf, ycf, t = self.predict(dataloader)
        # pred_0, pred_1, yf = scaler.reverse_y(pred_0), scaler.reverse_y(pred_1), scaler.reverse_y(yf)  # 标签反归一化
        mode = 'in-sample' if data == 'train' else 'out-sample'
        metric = metrics(pred_0, pred_1, yf, ycf, t, mode)

        return metric

def global_metric_update(metric: dict(), metric_: dict()):
    for key in metric_.keys():
        metric[key] = np.concatenate([metric[key], [metric_[key]]])
    return metric

if __name__ == "__main__":
    
    hparams = argparse.ArgumentParser(description='hparams')
    hparams.add_argument('--info', type=str, default='SITE')
    hparams.add_argument('--dim_pddm_in', type=int, default=60)
    hparams.add_argument('--dim_pddm_hide', type=int, default=32)
    hparams.add_argument('--dim_pddm_c', type=int, default=32)
    hparams.add_argument('--batchSize', type=int, default=32)
    hparams.add_argument('--reweight_sample', type=bool, default=False, help='whether or not to reweight samples based on propensity score')
    hparams.add_argument('--batch_norm', type=bool, default=False, help='whether or not to use batch normalization')
    hparams.add_argument('--varsel',type=bool, default=False,help='whether or not to use variable selection')

    hparams.add_argument('--model', type=str, default='ylearner')
    hparams.add_argument('--device', type=str, default='cuda:0')
    hparams.add_argument('--data', type=str, default='IHDP')
    hparams.add_argument('--epoch', type=int, default=400)
    hparams.add_argument('--seed', type=int, default=1)
    hparams.add_argument('--stop_epoch', type=int, default=30, help='tolerance epoch of early stopping')
    hparams.add_argument('--treat_weight', type=float, default=0.0, help='whether or not to balance sample')


    hparams.add_argument('--dim_backbone', type=str, default='60,60')
    hparams.add_argument('--dim_task', type=str, default='60,60')
    hparams.add_argument('--lr', type=float, default=1e-3)
    hparams.add_argument('--l2_reg', type=float, default=1e-4)
    hparams.add_argument('--dropout', type=float, default=0)
    hparams.add_argument('--treat_embed', type=bool, default=True)
    hparams.add_argument('--lambda_pddm', type=float, default=1.0, help='weight of pddm_loss in loss function')
    hparams.add_argument('--propensity_dir', type=str, default='simi_ite/tmp/propensity_model', help='directory storing the pretrained propensity artifacts')

    hparams.add_argument('--ot', type=str, default='uot', help='ot, uot, lot, none')
    hparams.add_argument('--lambda', type=float, default=2.0, help='weight of wass_loss in loss function')
    hparams.add_argument('--ot_scale', type=float, default=1.0, help='weight of x distance. In IHDP, it should be set to 0.5-2.0 according to simulation conditions')
    hparams.add_argument('--epsilon', type=float, default=1.0, help='Entropic Regularization in sinkhorn. In IHDP, it should be set to 0.5-5.0 according to simulation conditions')
    hparams.add_argument('--kappa', type=float, default=1.0, help='weight of marginal constraint in UOT. In IHDP, it should be set to 0.1-5.0 according to simulation conditions')
    hparams.add_argument('--gamma', type=float, default=0.0005, help='weight of joint distribution alignment. In IHDP, it should be set to 0.0001-0.005 according to simulation conditions')
    hparams.add_argument('--ot_joint_bp', type=bool, default=True, help='weight of joint distribution alignment')
    hparams_base = vars(hparams.parse_args())

    train_results = {
        "mae_ate": np.array([]), "mae_att": np.array([]), "pehe": np.array([]),
        "r2_f": np.array([]), "rmse_f": np.array([]), "r2_cf": np.array([]),
        "rmse_cf": np.array([]), "auuc": np.array([]), "rauuc": np.array([])
    }
    test_results = deepcopy(train_results)
    total_run = 10
    for i in range(total_run):
        print(f"Run {i+1}/{total_run} with seed {hparams_base['seed'] + i}")
        hparams = deepcopy(hparams_base)
        # 更新种子
        current_seed = hparams['seed'] + i 
        hparams['seed'] = current_seed
        # np.random.seed(current_seed)
        # # 如果用到了 cuda，最好也加上这个
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(current_seed)


        torch.manual_seed(current_seed)
        estimator = BaseEstimator(hparams=hparams)
        train_best_metric, test_best_metric = estimator.fit()
        train_results = global_metric_update(train_results, train_best_metric)
        test_results = global_metric_update(test_results, test_best_metric)
    print("Average Train Results over {} runs:".format(total_run))
    key_results = ["pehe",  "auuc"]
    for key in key_results:
        print(f"{key}: {np.mean(train_results[key])} ± {np.std(train_results[key])}")
    print("Average Test Results over {} runs:".format(total_run))
    for key in key_results:
        print(f"{key}: {np.mean(test_results[key])} ± {np.std(test_results[key])}")
        # writer.close()
