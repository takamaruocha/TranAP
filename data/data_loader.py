import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class Dataset_MTS(Dataset):
    def __init__(self, data, root_path, step_size, flag='train', size=None, data_split = [0.8, 0.2], scale=True):
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        self.step_size = step_size
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale

        self.data = data
        self.root_path = root_path

        self.data_split = data_split
        self.__read_data__()

    def __read_data__(self):
        data_folder = self.root_path + self.data + '/'
        arr_raw = np.load(os.path.join(data_folder, 'train.npy'), allow_pickle=True)
        df_raw = pd.DataFrame(arr_raw)
        df_raw = df_raw.values

        arr_raw_test = np.load(os.path.join(data_folder, 'test.npy'), allow_pickle=True)
        df_raw_test = pd.DataFrame(arr_raw_test)
        df_raw_test = df_raw_test.values

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(df_raw)
            df_raw = self.scaler.transform(df_raw)
            df_raw_test = self.scaler.transform(df_raw_test)

        arr_raw_test_label = np.load(os.path.join(data_folder, 'test_label.npy'), allow_pickle=True)
        df_raw_test_label = pd.DataFrame(arr_raw_test_label)

        train_num = int(len(df_raw)*self.data_split[0]); 
        val_num = len(df_raw) - train_num; 
        
        border1s = [0, train_num]
        border2s = [train_num, train_num+val_num]

        if self.flag == 'train':
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            self.data_x = self.data_y = df_raw[border1:border2]

        elif self.flag == 'val':
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            self.data_x = self.data_y = df_raw[border1:border2]

        else:
            self.data_x = self.data_y = df_raw_test
            self.test_label = df_raw_test_label.values
    
    def __getitem__(self, index):
        index = index * self.step_size
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len

        if self.flag == 'test':
            return self.data_x[s_begin:s_end], self.data_y[r_begin:r_end], self.test_label[r_begin:r_end]
        else:
            return self.data_x[s_begin:s_end], self.data_y[r_begin:r_end]   

    def __len__(self):
        return (len(self.data_x) - self.in_len + self.step_size - self.out_len) // self.step_size

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class PSMSegLoader(Dataset):
    def __init__(self, data, root_path, step_size, flag='train', size=None, data_split = [0.8, 0.2], scale=True):
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        self.step_size = step_size
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.data = data
        self.root_path = root_path

        self.data_split = data_split
        self.__read_data__()

    def __read_data__(self):
        data_folder = self.root_path + self.data + '/'
        df_raw = pd.read_csv(os.path.join(data_folder, 'train.csv'))
        df_raw = df_raw.values[:, 1:]
        df_raw = np.nan_to_num(df_raw)

        df_raw_test = pd.read_csv(os.path.join(data_folder, 'test.csv'))
        df_raw_test = df_raw_test.values[:, 1:]
        df_raw_test = np.nan_to_num(df_raw_test)

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(df_raw)
            df_raw = self.scaler.transform(df_raw)
            df_raw_test = self.scaler.transform(df_raw_test)

        df_raw_test_label = pd.read_csv(os.path.join(data_folder, 'test_label.csv'))
        df_raw_test_label = df_raw_test_label.values[:, 1:]

        train_num = int(len(df_raw)*self.data_split[0]);
        val_num = len(df_raw) - train_num;

        border1s = [0, train_num]
        border2s = [train_num, train_num+val_num]

        if self.flag == 'train':
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            self.data_x = self.data_y = df_raw[border1:border2]

        elif self.flag == 'val':
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            self.data_x = self.data_y = df_raw[border1:border2]

        else:
            self.data_x = self.data_y = df_raw_test
            self.test_label = df_raw_test_label

    def __getitem__(self, index):
        index = index * self.step_size
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len

        if self.flag == 'test':
            return self.data_x[s_begin:s_end], self.data_y[r_begin:r_end], self.test_label[r_begin:r_end]
        else:
            return self.data_x[s_begin:s_end], self.data_y[r_begin:r_end]

    def __len__(self):
        return (len(self.data_x) - self.in_len + self.step_size - self.out_len) // self.step_size

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class SWaTSegLoader(Dataset):
    def __init__(self, data, root_path, step_size, flag='train', size=None, data_split = [0.8, 0.2], scale=True):
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        self.step_size = step_size
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.data = data
        self.root_path = root_path

        self.data_split = data_split
        self.__read_data__()

    def __read_data__(self):
        data_folder = self.root_path + self.data + '/'
        df_raw = pd.read_csv(os.path.join(data_folder, 'swat_train2.csv'))
        df_raw = df_raw.values[:, :-1]

        df_raw_test = pd.read_csv(os.path.join(data_folder, 'swat2.csv'))
        df_raw_test_label = df_raw_test.values[:, -1:]
        df_raw_test = df_raw_test.values[:, :-1]

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(df_raw)
            df_raw = self.scaler.transform(df_raw)
            df_raw_test = self.scaler.transform(df_raw_test)

        train_num = int(len(df_raw)*self.data_split[0]);
        val_num = len(df_raw) - train_num;

        border1s = [0, train_num]
        border2s = [train_num, train_num+val_num]

        if self.flag == 'train':
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            self.data_x = self.data_y = df_raw[border1:border2]

        elif self.flag == 'val':
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            self.data_x = self.data_y = df_raw[border1:border2]

        else:
            self.data_x = self.data_y = df_raw_test
            self.test_label = df_raw_test_label

    def __getitem__(self, index):
        index = index * self.step_size
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len

        if self.flag == 'test':
            return self.data_x[s_begin:s_end], self.data_y[r_begin:r_end], self.test_label[r_begin:r_end]
        else:
            return self.data_x[s_begin:s_end], self.data_y[r_begin:r_end]

    def __len__(self):
        return (len(self.data_x) - self.in_len + self.step_size - self.out_len) // self.step_size

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def get_loader_segment(data, root_path, step_size, flag='train', size=None, data_split = [0.8, 0.2]):
    if (data == 'SWaT'):
        dataset = SWaTSegLoader(data, root_path, step_size, flag, size, data_split, scale=True)
    elif (data == 'PSM'):
        dataset = PSMSegLoader(data, root_path, step_size, flag, size, data_split, scale=True)
    else: #if (data == 'SMD' or data == 'SMAP' or data == 'NIPS_TS_Water'):
        dataset = Dataset_MTS(data, root_path, step_size, flag, size, data_split, scale=True)

    return dataset

