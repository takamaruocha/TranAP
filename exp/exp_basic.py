import os
import torch
import torch.nn as nn
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        super(Exp_Basic, self).__init__()
        self.args = args
        self.device = self._acquire_device()
        self.model1, self.model2 = self._build_model()
        self.model1.to(self.device)
        self.model2.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def training(self):
        pass

    def test(self):
        pass
