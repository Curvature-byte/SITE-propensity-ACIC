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
from simi_ite.data_processor import MyDataset
from models import *
from simi_ite.utils import find_three_pairs, get_simi_ground, metric_update, get_three_pair_simi, similarity_score, metric_export,metrics
from simi_ite.pddm_net import compute_pddm_loss, PDDMNetwork
from simi_ite.propensity import load_propensity_score




class BaseEstimator:

    def __init__(self, board=None, hparams={}):
        data_name = hparams.get('data')
        print("Current data:", data_name)
        train_set = MyDataset(f"Datasets/{data_name}/train.csv", fit=True)
        shared_scaler = train_set.scaler
        traineval_set = MyDataset(f"Datasets/{data_name}/traineval.csv", scaler=shared_scaler)
        eval_set = MyDataset(f"Datasets/{data_name}/eval.csv", scaler=shared_scaler)
        test_set = MyDataset(f"Datasets/{data_name}/test.csv", scaler=shared_scaler)
        self.device = torch.device(hparams.get('device'))
        if hparams['treat_weight'] == 0:
            self.train_loader = DataLoader(train_set, batch_size=hparams.get('batchSize'), drop_last=True,shuffle=True)
        else:
            self.train_loader = DataLoader(train_set, batch_size=hparams.get('batchSize'), sampler=train_set.get_sampler(hparams['treat_weight']), drop_last=True)
        self.traineval_data = DataLoader(traineval_set, batch_size=256)  # for test in-sample metric
        self.eval_data = DataLoader(eval_set, batch_size=256,shuffle=True)
        self.test_data = DataLoader(test_set, batch_size=256,shuffle=True)

        self.scaler = shared_scaler
        self.train_scaler = shared_scaler
        self.eval_scaler = shared_scaler
        self.test_scaler = shared_scaler

        self.train_metric = {
             "r2_f": np.array([]),
             "rmse_f": np.array([]),
             "mae_f": np.array([]),
            }
        self.eval_metric = deepcopy(self.train_metric)
        self.test_metric = deepcopy(self.train_metric)

        self.train_best_metric = {
             "r2_f": None,
             "rmse_f": None,
             "mae_f": None,
            }
        self.eval_best_metric = deepcopy(self.train_best_metric)
        self.eval_best_metric['r2_f'] = -10  
        self.loss_metric = {'loss': np.array([]), 'loss_f': np.array([]), 'loss_pddm': np.array([])}

        self.epochs = hparams.get('epoch', 200)
        self._pddm_warning_emitted = False

        self.model = TARNetHead(train_set.x_dim - 1, hparams).to(self.device)
        self.rep_model = self.model.rep
        self.propensity_model = load_propensity_score(model_dir = './simi_ite/tmp/propensity_model' ,input_dim = train_set.x_dim - 1 ,mode="mlp")
        self.pddm_model = PDDMNetwork(
            dim_in=hparams['dim_pddm_in'],
            dim_pddm=hparams['dim_pddm_hide'],
            dim_c=hparams['dim_pddm_c'],
        ).to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.pddm_model.parameters()),
            lr=hparams.get('lr', 1e-3),
            weight_decay=hparams.get('l2_reg', 1e-4)
        )
        self.hparams = hparams
        self.epoch = 0
        self.board = board

    def fit(self):

        iter_num = 0
        stop_epoch = 0 # record how many iterations the eval metrics do not improve
        for epoch in tqdm(range(1, self.epochs)):

            self.epoch = epoch
            self.model.train()
            self.pddm_model.train()
            for data in self.train_loader:  # train_loader
                self.optimizer.zero_grad()
                data = data.to(self.device)
                _x, _xt, _t, _yf = data[:, 1:-3], data[:, 1:-2], data[:, -3], data[:, -1]
                _pred_f = self.model(_xt).squeeze(-1)
                _loss_fit = self.criterion(_pred_f, _yf)

                treated = int((_t > 0.5).sum().item())
                control = int((_t < 0.5).sum().item())
                pddm_indicator = (epoch >20 and treated >= 2 and control >= 2)

                loss_pddm_tensor = torch.tensor(0.0, device=self.device)
                loss_pddm_value = 0.0

                if pddm_indicator:
                    x_np = _x.detach().cpu().numpy()
                    t_np = _t.detach().cpu().numpy()
                    _x_propensity_scores, _x_simi_ground = get_simi_ground(x_np, propensity_model=self.propensity_model)
                    try:
                        three_pairs, three_paris_index = find_three_pairs(x_np, t_np, _x_propensity_scores)
                    except ValueError as exc:
                        if not self._pddm_warning_emitted:
                            print(f"[SITE] Skip PDDM for this batch: {exc}")
                            self._pddm_warning_emitted = True
                    else:
                        three_pairs = torch.tensor(three_pairs, dtype=torch.float32, device=self.device)
                        three_pairs_rep = self.rep_model(three_pairs)
                        loss_pddm_tensor = compute_pddm_loss(
                            self.pddm_model,
                            three_pairs_rep,
                            _x_simi_ground,
                            three_paris_index
                        ).squeeze()
                        # loss_pddm_value = loss_pddm_tensor.item()

                _loss = _loss_fit + self.hparams['lambda'] * loss_pddm_tensor
                _loss.backward()
                self.optimizer.step()
                # _x_propensity_score, _x_similarity_ground = get_simi_ground(_x.detach().cpu().numpy(),  propensity_model=self.propensity_model)
                # three_pairs , three_paris_index = find_three_pairs(_x.detach().cpu().numpy(), _t.detach().cpu().numpy(), _x_propensity_score)
                # three_pairs = torch.tensor(three_pairs, dtype=torch.float32,device=self.device)
                # _pred_f = self.model(_xt)
                # three_rep_paris = self.rep_model(three_pairs)
                
                # # Section: loss calculation
                # _loss_fit = self.criterion(_pred_f, _yf)

                # _loss_pddm = 0
                # pddm_indicator = ( epoch > 20 and len(_t.unique()) > 1)
                # if pddm_indicator: # Avoid samples coming from same group
                #     _loss_pddm = compute_pddm_loss(
                #         self._loss_pddm_model,
                #         three_rep_paris,
                #         _x_similarity_ground, three_paris_index
                #     ).squeeze()

                # _loss = _loss_fit + self.hparams['lambda'] * _loss_pddm
                # _loss.backward()
                # self.optimizer.step()

                # Section: metric update
                # _loss_pddm = loss_pddm_tensor.item() if pddm_indicator else 0
                self.board.add_scalar(
                    'loss/fit_loss',
                    _loss_fit.item(),
                    global_step=iter_num)
                self.board.add_scalar(
                    'loss/pddm_loss',
                    loss_pddm_tensor.item(),
                    global_step=iter_num)
                self.board.add_scalar(
                    'loss/total_loss',
                    _loss.item(),
                    global_step=iter_num)
                iter_num += 1

            # Section: evaluation and model selection
            if self.epoch % 10 == 0:
                _train_metric = self.evaluation(data='train')
                self.train_metric = metric_update(self.train_metric, _train_metric, self.epoch)
                [self.board.add_scalar(f"train/{key}", _train_metric[key], global_step=self.epoch) for key in _train_metric.keys()]

            if self.epoch % 1 == 0:
                _eval_metric = self.evaluation(data='eval')
                self.eval_metric = metric_update(self.eval_metric, _eval_metric, self.epoch)
                [self.board.add_scalar(f"eval/{key}", _eval_metric[key], global_step=self.epoch) for key in _eval_metric.keys()]

                if _eval_metric['r2_f'] > self.eval_best_metric['r2_f']:
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

        # _test_metric = self.evaluation(data='test')
        # self.test_metric = metric_update(self.test_metric, _test_metric, self.epoch, )
        # [self.board.add_scalar(f"test/{key}", _test_metric[key], global_step=self.epoch) for key in _test_metric.keys()]

        [self.board.add_scalar(f"train_best/{key}", self.train_best_metric[key], global_step=self.epoch) for key in self.train_best_metric.keys()]
        [self.board.add_scalar(f"eval_best/{key}", self.eval_best_metric[key], global_step=self.epoch) for key in self.eval_best_metric.keys()]
        [self.board.add_scalar(f"test_best/{key}", self.test_best_metric[key], global_step=self.epoch) for key in self.test_best_metric.keys()]
        # self.eval_best_metric = {'eval_best/' + key: self.eval_best_metric[key] for key in self.eval_best_metric.keys()}
        # self.board.add_hparams(hparam_dict=self.hparams, metric_dict=self.eval_best_metric)
        self.board.add_graph(self.model, _xt)
        metric_export(path, self.train_best_metric, self.eval_best_metric, self.test_best_metric)

    def predict(self, dataloader):
        """

        :param dataloader
        :return: np.array, shape: (#sample)
        """
        self.model.eval()
        pred_0 = torch.tensor([], device=self.device)
        yf = deepcopy(pred_0)
        # pred_1, yf, ycf, t = deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0),

        for data in dataloader:
            data = data.to(self.device)
            _xt, _yf = data[:, 1:-2], data[:, -2]
            # _x, _t, _yf, _ycf = data[:, :-3], data[:, [-3]], data[:, -2], data[:, -1]
            # _x_0 = torch.cat([_x, torch.zeros_like(_t, device=self.device)], dim=-1)
            # _x_1 = torch.cat([_x, torch.ones_like(_t, device=self.device)], dim=-1)
            _pred_0 = self.model(_xt).reshape([-1])
            # _pred_1 = self.model(_x_1).reshape([-1])
            pred_0 = torch.cat([pred_0, _pred_0], axis=-1)
            # pred_1 = torch.cat([pred_1, _pred_1], axis=-1)
            yf = torch.cat([yf, _yf], axis=-1)
            # ycf = torch.cat([ycf, _ycf], axis=-1)
            # t = torch.cat([t, _t.reshape([-1])], axis=-1)

        pred_0 = pred_0.detach().cpu().numpy()
        # pred_1 = pred_1.detach().cpu().numpy()
        yf = yf.cpu().numpy()
        pred_0 = self.scaler.reverse_y(pred_0)
        yf = self.scaler.reverse_y(yf)
        # ycf = ycf.cpu().numpy()
        # t = t.detach().cpu().numpy()
        return pred_0, yf # pred_1, yf, ycf, t

    def evaluation(self, data: str) -> dict():

        dataloader = {
            'train': self.traineval_data,
            'eval': self.eval_data,
            'test': self.test_data}[data]
        pred_0, yf = self.predict(dataloader)
        # pred_0, pred_1, yf, ycf, t = self.predict(dataloader)
        # pred_0, pred_1, yf = scaler.reverse_y(pred_0), scaler.reverse_y(pred_1), scaler.reverse_y(yf)  # 标签反归一化
        # mode = 'in-sample' if data == 'train' else 'out-sample'
        # metric = metrics(pred_0, pred_1, yf, ycf, t, mode)
        metric = metrics(pred_0, yf)

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
    hparams.add_argument('--data', type=str, default='ACIC')
    hparams.add_argument('--epoch', type=int, default=30)
    hparams.add_argument('--seed', type=int, default=42)
    hparams.add_argument('--stop_epoch', type=int, default=30, help='tolerance epoch of early stopping')
    hparams.add_argument('--treat_weight', type=float, default=0.0, help='whether or not to balance sample')


    hparams.add_argument('--dim_backbone', type=str, default='60,60')
    hparams.add_argument('--dim_task', type=str, default='60,60')
    hparams.add_argument('--lr', type=float, default=1e-3)
    hparams.add_argument('--l2_reg', type=float, default=1e-4)
    hparams.add_argument('--dropout', type=float, default=0)
    hparams.add_argument('--treat_embed', type=bool, default=True)
    hparams.add_argument('--lambda', type=float, default=0, help='weight of wass_loss in loss function')
    hparams.add_argument('--propensity_dir', type=str, default='simi_ite/tmp/propensity_model', help='directory storing the pretrained propensity artifacts')
    hparams = vars(hparams.parse_args())

    # path = f"Resultsparam/{hparams['data']}/{hparams['model']}/{hparams['ot']}_{hparams['lambda']}_{hparams['epsilon']}_{hparams['kappa']}_{hparams['gamma']}_{hparams['batchSize']}_{hparams['treat_weight']}_{hparams['ot_scale']}_{hparams['seed']}"
    path = f"Resultsparam/{hparams['data']}/{hparams['model']}/SITE_{hparams['lambda']}_{hparams['batchSize']}_{hparams['treat_weight']}_{hparams['seed']}" 
    writer = SummaryWriter(path)
    torch.manual_seed(hparams['seed'])
    if hasattr(os, 'nice'):
        os.nice(0)
    estimator = BaseEstimator(board=writer, hparams=hparams)
    estimator.fit()
    writer.close()
