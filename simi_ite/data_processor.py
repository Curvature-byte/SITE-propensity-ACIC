import kagglehub
import pandas as pd
import numpy as np
import torch
import pyreadr
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from simi_ite.utils import StandardScaler as SiteStandardScaler

class MyDataset(Dataset):

    def __init__(self, path, mask=0, scaler=None, fit=False):
        self.data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        self.x_dim = self.data.shape[-1] - 3
        self.x_dim_start = int(mask * self.x_dim)
        self.x_dim -= self.x_dim_start
        self.scaler = scaler or SiteStandardScaler(protected_indices=[-3])
        if fit or scaler is None:
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.data = self.scaler.transform(self.data)

    def __getitem__(self, index):

        return self.data[index, self.x_dim_start:]

    def __len__(self):

        return len(self.data)

    def get_sampler(self, treat_weight=1):

        t = self.data[:, -3].astype(np.int16)
        count = Counter(t)
        class_count = np.array([count[0], count[1]*treat_weight])
        weight = 1. / class_count
        samples_weight = torch.tensor([weight[item] for item in t])
        sampler = WeightedRandomSampler(
            samples_weight,
            len(samples_weight),
            replacement=True)

        return sampler







def TEP_processor(random_state=1):

    # Download latest version
    path = kagglehub.dataset_download("averkij/tennessee-eastman-process-simulation-dataset")

    print("Path to dataset files:", path)
    # Load the Rtrain_Data file
    result = pyreadr.read_r("/home/jiangwei/.cache/kagglehub/datasets/averkij/tennessee-eastman-process-simulation-dataset/versions/1/TEP_FaultFree_Training.RData")
    result1 = pyreadr.read_r("/home/jiangwei/.cache/kagglehub/datasets/averkij/tennessee-eastman-process-simulation-dataset/versions/1/TEP_Faulty_Training.RData")
    result2 = pyreadr.read_r("/home/jiangwei/.cache/kagglehub/datasets/averkij/tennessee-eastman-process-simulation-dataset/versions/1/TEP_FaultFree_Testing.RData")
    result3 = pyreadr.read_r("/home/jiangwei/.cache/kagglehub/datasets/averkij/tennessee-eastman-process-simulation-dataset/versions/1/TEP_Faulty_Testing.RData")
    # Extract the train_data frame
    df_normal_train = result['fault_free_training']
    df_fault_train = result1['faulty_training']
    df_normal_test = result2['fault_free_testing']
    df_fault_test = result3['faulty_testing']

    df_normal_train.head(10)
    df_fault_train.head(10)

    X_dict = {
    'XMEAS_1':'A_feed_stream',
    'XMEAS_2':'D_feed_stream',
    'XMEAS_3':'E_feed_stream',
    'XMEAS_4':'Total_fresh_feed_stripper',
    'XMEAS_5':'Recycle_flow_into_rxtr',
    'XMEAS_6':'Reactor_feed_rate',
    'XMEAS_7':'Reactor_pressure',
    'XMEAS_8':'Reactor_level',
    'XMEAS_9':'Reactor_temp',
    'XMEAS_10':'Purge_rate',
    'XMEAS_11':'Separator_temp',
    'XMEAS_12':'Separator_level',
    'XMEAS_13':'Separator_pressure',
    'XMEAS_14':'Separator_underflow',
    'XMEAS_15':'Stripper_level',
    'XMEAS_16':'Stripper_pressure',
    'XMEAS_17':'Stripper_underflow',
    'XMEAS_18':'Stripper_temperature',
    'XMEAS_19':'Stripper_steam_flow',
    'XMEAS_20':'Compressor_work',
    'XMEAS_21':'Reactor_cooling_water_outlet_temp',
    'XMEAS_22':'Condenser_cooling_water_outlet_temp',
    'XMEAS_23':'Composition_of_A_rxtr_feed',
    'XMEAS_24':'Composition_of_B_rxtr_feed',
    'XMEAS_25':'Composition_of_C_rxtr_feed',
    'XMEAS_26':'Composition_of_D_rxtr_feed',
    'XMEAS_27':'Composition_of_E_rxtr_feed',
    'XMEAS_28':'Composition_of_F_rxtr_feed',
    'XMEAS_29':'Composition_of_A_purge',
    'XMEAS_30':'Composition_of_B_purge',
    'XMEAS_31':'Composition_of_C_purge',
    'XMEAS_32':'Composition_of_D_purge',
    'XMEAS_33':'Composition_of_E_purge',
    'XMEAS_34':'Composition_of_F_purge',
    'XMEAS_35':'Composition_of_G_purge',
    'XMEAS_36':'Composition_of_H_purge',
    'XMEAS_37':'Composition_of_D_product',
    'XMEAS_38':'Composition_of_E_product',
    'XMEAS_39':'Composition_of_F_product',
    'XMEAS_40':'Composition_of_G_product',
    'XMEAS_41':'Composition_of_H_product',
    'XMV_1':'D_feed_flow_valve',
    'XMV_2':'E_feed_flow_valve',
    'XMV_3':'A_feed_flow_valve',
    'XMV_4':'Total_feed_flow_stripper_valve',
    'XMV_5':'Compressor_recycle_valve',
    'XMV_6':'Purge_valve',
    'XMV_7':'Separator_pot_liquid_flow_valve',
    'XMV_8':'Stripper_liquid_product_flow_valve',
    'XMV_9':'Stripper_steam_valve',
    'XMV_10':'Reactor_cooling_water_flow_valve',
    'XMV_11':'Condenser_cooling_water_flow_valve',
    'XMV_12':'Agitator_speed'
    }
    df_normal_train= df_normal_train.rename(columns = lambda x:X_dict[x.upper()] if x.upper() in X_dict.keys()  else x)
    df_fault_train = df_fault_train.rename(columns = lambda x:X_dict[x.upper()] if x.upper() in X_dict.keys()  else x)
    df_normal_test= df_normal_test.rename(columns = lambda x:X_dict[x.upper()] if x.upper() in X_dict.keys()  else x)
    df_fault_test = df_fault_test.rename(columns = lambda x:X_dict[x.upper()] if x.upper() in X_dict.keys()  else x)
    # Create training set
    train_data = pd.concat([df_normal_train, df_fault_train[df_fault_train['faultNumber']==1]], axis=0)
    train_data['T'] = np.where((train_data['faultNumber'] == 1) & (train_data['sample'] > 20), 1, 0)
    train_data['y1'] = train_data['Composition_of_G_product']
    train_data['y2'] = train_data['Composition_of_H_product']
    # Create evaluation and test sets
    eval_test_data = pd.concat([df_normal_test, df_fault_test[df_fault_test['faultNumber']==1]], axis=0)
    eval_test_data['T'] = np.where((eval_test_data['faultNumber'] == 1) & (eval_test_data['sample'] > 180), 1, 0)
    eval_test_data['y1'] = eval_test_data['Composition_of_G_product']
    eval_test_data['y2'] = eval_test_data['Composition_of_H_product']


    train_data = train_data.drop(['faultNumber', 'simulationRun', "Composition_of_G_product", "Composition_of_H_product"], axis=1)
    eval_test_data = eval_test_data.drop(['faultNumber', 'simulationRun', "Composition_of_G_product", "Composition_of_H_product"], axis=1)
    train_data = train_data.to_numpy()
    eval_test_data = eval_test_data.to_numpy()
    evaluation_data, test_data = train_test_split(eval_test_data, test_size=0.27, stratify=eval_test_data[:, -3], random_state=random_state)
    np.savetxt("Datasets/TEP/train.csv", train_data, delimiter=",")
    np.savetxt("Datasets/TEP/traineval.csv", np.concatenate((train_data, evaluation_data), axis=0), delimiter=",")
    np.savetxt("Datasets/TEP/eval.csv", evaluation_data, delimiter=",")
    np.savetxt("Datasets/TEP/test.csv", test_data, delimiter=",")
    print('New TEP train_Data')
    return None



if __name__ == "__main__":

    # acic2016_processor()
    # [ihdp_processor(i) for i in range(1,11)]
    TEP_processor()