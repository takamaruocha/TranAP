import numpy
import os
import time
import json
import pickle

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from data.data_loader import get_loader_segment
from exp.exp_basic import Exp_Basic
from models.transformer_tf_attn import Transformer_TF_ATTN

from utilities.tools import EarlyStopping, adjust_learning_rate
from utilities.metrics import metric

from src.pot import *
from pprint import pprint

import warnings
warnings.filterwarnings('ignore')

class Exp_TranAP(Exp_Basic):
    def __init__(self, args):
        super(Exp_TranAP, self).__init__(args)

    def _build_model(self):
        '''''''''''''''''''''''''''
        Transformer for forecasitng
        '''''''''''''''''''''''''''
        model1 = Transformer_TF_ATTN(
            self.args.data_dim,
            self.args.in_len,
            self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model,
            self.args.d_ff,
            self.args.n_heads,
            self.args.e_layers,
            self.args.dropout,
            self.args.attn_ratio,
            self.device
        ).float().to(self.device)

        '''''''''''''''''''''''''''
        Transformer for reconstruction
        '''''''''''''''''''''''''''
        model2 = Transformer_TF_ATTN(
            self.args.data_dim,
            self.args.in_len+self.args.out_len,
            self.args.in_len+self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model,
            self.args.d_ff,
            self.args.n_heads,
            self.args.e_layers,
            self.args.dropout,
            self.args.attn_ratio,
            self.device
        ).float().to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model1 = nn.DataParallel(model1, device_ids=self.args.device_ids)
            model2 = nn.DataParallel(model2, device_ids=self.args.device_ids)

        return model1, model2

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
       
        data_set = get_loader_segment(
            data = args.data,
            root_path = args.root_path,
            step_size = args.step_size,
            flag = flag,
            size = [args.in_len, args.out_len],
            data_split = args.data_split
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size = batch_size,
            shuffle = shuffle_flag,
            num_workers = 0,
            drop_last = drop_last
        )

        return data_set, data_loader

    def _select_optimizer(self, model):
        model_optim = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, reduction='mean'):
        criterion =  nn.MSELoss(reduction=reduction)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model1.eval(), self.model2.eval()
        total_loss_tsf = []
        total_loss_tsr = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch_tsf(batch_x, batch_y)
                predict_data = torch.cat((batch_x.to(self.device), pred), 1)
                loss_tsf = criterion(pred, true)
                total_loss_tsf.append(loss_tsf.item())

                rec, _ = self._process_one_batch_tsr(predict_data)
                loss_tsr = criterion(rec, predict_data)
                total_loss_tsr.append(loss_tsr.item())

        total_loss_tsf = np.average(total_loss_tsf)
        total_loss_tsr = np.average(total_loss_tsr)
        self.model1.train(), self.model2.train()

        return total_loss_tsf, total_loss_tsr

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model1_optim = self._select_optimizer(self.model1)
        model2_optim = self._select_optimizer(self.model2)
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss_tsf = []
            train_loss_tsr = []

            self.model1.train(), self.model2.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1

                pred, true = self._process_one_batch_tsf(batch_x, batch_y)
                loss_tsf = criterion(pred, true)
                train_loss_tsf.append(loss_tsf.item())

                model1_optim.zero_grad()
                loss_tsf.backward()
                model1_optim.step()
                
                target = torch.cat((batch_x, batch_y), 1).float()
                rec, _  = self._process_one_batch_tsr(target)
                loss_tsr = criterion(rec, target.to(self.device))
                train_loss_tsr.append(loss_tsr.item())

                model2_optim.zero_grad()
                loss_tsr.backward()
                model2_optim.step()

                if (i+1) % 100==0:
                    print('iters: {0}, epoch: {1} | loss_tsf: {2:.3f} | loss_rec: {3:.3f}'.format(i+1, epoch+1, loss_tsf.item(), loss_tsr.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print('Epoch: {0} cost time: {1}'.format(epoch + 1, time.time() - epoch_time))
            train_loss_tsf_avg = np.average(train_loss_tsf)
            train_loss_tsr_avg = np.average(train_loss_tsr)
            vali_loss_tsf, vali_loss_tsr = self.vali(vali_data, vali_loader, criterion)

            print('Epoch: {0}, Steps: {1} | Train Loss_TSF: {2:.3f} Vali Loss_TSF: {3:.3f} | Train Loss_REC: {4:.3f} Vali Loss_REC: {5:.3f}'.format(
                epoch+1, train_steps, train_loss_tsf_avg, vali_loss_tsf, train_loss_tsr_avg, vali_loss_tsr))
            
            early_stopping(vali_loss_tsf, vali_loss_tsr, self.model1, self.model2, path)
            if early_stopping.early_stop:
                print('Early stopping')
                break

            adjust_learning_rate(model1_optim, model2_optim, epoch+1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model1.load_state_dict(torch.load(best_model_path)['model1'])
        self.model2.load_state_dict(torch.load(best_model_path)['model2'])
        state_dict = self.model1.module.state_dict() if isinstance(self.model1, DataParallel) else self.model1.state_dict()
        state_dict2 = self.model2.module.state_dict() if isinstance(self.model2, DataParallel) else self.model2.state_dict()
        
        torch.save({
                    'model1': state_dict,
                    'model2': state_dict2
                   }, path + '/' + 'checkpoint.pth')

        return self.model1, self.model2


    def test(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        criterion =  self._select_criterion(reduction='none')

        self.model1.eval(), self.model2.eval()

        metrics_all = []
        instance_num = 0

        scores_train = []
        scores_test = []
        labels = []
        start_time = time.time()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                # Compute anomaly prediction score of training dataset
                pred, true = self._process_one_batch_tsf(batch_x, batch_y)
                predict_data = torch.cat((batch_x.to(self.device), pred), 1)
                rec, pred = self._process_one_batch_tsr(predict_data)
                score = criterion(rec, predict_data)
                scores_train.append(np.sum(score.detach().cpu().numpy(), (1,2)))

            for i, (batch_x, batch_y, batch_label) in enumerate(test_loader):
                # Anomaly label
                batch_label = (np.sum(batch_label.detach().cpu().numpy(), (1,2)) >= 1) + 0
                labels.append(batch_label)
                
                # Compute anomaly prediction score of test dataset
                pred, true = self._process_one_batch_tsf(batch_x, batch_y)
                predict_data = torch.cat((batch_x.to(self.device), pred), 1)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)

                rec, pred = self._process_one_batch_tsr(predict_data)
                score = criterion(rec, predict_data)
                scores_test.append(np.sum(score.detach().cpu().numpy(), (1,2)))

        scores_train = np.concatenate(scores_train)
        scores_test = np.concatenate(scores_test)
        labels = np.concatenate(labels)
        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # Perform anomaly prediction
        p_t, _ = pot_eval(scores_train, scores_test, labels, self.args.data)
        print('TEST | cost time: {}'.format(time.time() - start_time))
        print('TEST | F1: {0:.3f} | PRE: {1:.3f} | REC: {2:.3f} | TP: {3} | TN: {4} | FP: {5} | FN: {6} | ROC/AUC: {7:.3f} | FPR: {8:.3f}'.format(p_t[0], p_t[1], p_t[2], p_t[3], p_t[4], p_t[5], p_t[6], p_t[7], p_t[8]))

        # Compute MAE and MSE
        mae, mse, _, _, _ = metrics_mean
        print('MSE:{0:.3f}, MAE:{1:.3f}'.format(mse, mae))

        return p_t[0], p_t[1], p_t[2]

    def eval(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        criterion =  self._select_criterion(reduction='none')

        self.model1.eval(), self.model2.eval()

        metrics_all = []
        instance_num = 0

        scores_train = []
        scores_test = []
        labels = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                # Compute anomaly prediction score of training dataset
                pred, true = self._process_one_batch_tsf(batch_x, batch_y)
                predict_data = torch.cat((batch_x.to(self.device), pred), 1)
                rec, pred = self._process_one_batch_tsr(predict_data)
                score = criterion(rec, predict_data)
                scores_train.append(np.sum(score.detach().cpu().numpy(), (1,2)))

            for i, (batch_x, batch_y, batch_label) in enumerate(test_loader):
                # Anomaly label
                batch_label = (np.sum(batch_label.detach().cpu().numpy(), (1,2)) >= 1) + 0
                labels.append(batch_label)

                # Compute anomaly prediction score of test dataset
                pred, true = self._process_one_batch_tsf(batch_x, batch_y)
                predict_data = torch.cat((batch_x.to(self.device), pred), 1)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)

                rec, pred = self._process_one_batch_tsr(predict_data)
                score = criterion(rec, predict_data)
                scores_test.append(np.sum(score.detach().cpu().numpy(), (1,2)))

        scores_train = np.concatenate(scores_train)
        scores_test = np.concatenate(scores_test)
        labels = np.concatenate(labels)
        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # Perform anomaly prediction
        p_t, _ = pot_eval(scores_train, scores_test, labels, self.args.data)
        print('TEST | F1: {0:.3f} | PRE: {1:.3f} | REC: {2:.3f} | TP: {3} | TN: {4} | FP: {5} | FN: {6} | ROC/AUC: {7:.3f} | FPR: {8:.3f}'.format(p_t[0], p_t[1], p_t[2], p_t[3], p_t[4], p_t[5], p_t[6], p_t[7], p_t[8]))

        # Compute MAE and MSE
        mae, mse, _, _, _ = metrics_mean
        print('TEST | MSE:{0:.3f}, MAE:{1:.3f}'.format(mse, mae))

        return p_t[0], p_t[1], p_t[2], mae, mse

    # Transformer for forecasting
    def _process_one_batch_tsf(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model1(batch_x)

        return outputs, batch_y

    # Tranformer for reconstruction
    def _process_one_batch_tsr(self, batch):
        batch = batch.float().to(self.device)

        outputs = self.model2(batch)

        return outputs, batch
