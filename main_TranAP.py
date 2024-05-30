import argparse
import os
import torch
import statistics

from exp.exp_TranAP import Exp_TranAP
from utilities.tools import string_split

parser = argparse.ArgumentParser(description='TranAP')

parser.add_argument('--data', type=str, required=True, default='SWaT', help='data')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/TranAP/', help='location to store model checkpoints')

parser.add_argument('--in_len', type=int, default=48, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=24, help='output(prediction) MTS length (tau)')
parser.add_argument('--step_size', type=int, default=6, help='step size')
parser.add_argument('--seg_len', type=int, default=12, help='segment length (L_seg)')
parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
parser.add_argument('--factor', type=int, default=10, help='num of routers')

parser.add_argument('--data_dim', type=int, default=7, help='dimensions of the MTS data (D)')
parser.add_argument('--d_model', type=int, default=256, help='dimension of hidden states (d_model)')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of feedforward network in transformer (d_ff)')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--attn_ratio', type=float, default=0.25, help='attention ratio (lambda)')

parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--itr', type=int, default=5, help='experiments times')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

data_parser = {
    'SWaT':{'data_dim':51, 'split':[0.8, 0.2]},
    'PSM':{'data_dim':25, 'split':[0.8, 0.2]},
    'SMD':{'data_dim':38, 'split':[0.8, 0.2]},
    'SMAP':{'data_dim':25, 'split':[0.8, 0.2]},
    'NIPS_TS_Water':{'data_dim':9, 'split':[0.8, 0.2]},
    }

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_dim = data_info['data_dim']
    args.data_split = data_info['split']
else:
    args.data_split = string_split(args.data_split)

print('Args in experiment:')
print(args)

Exp = Exp_TranAP

f1s = []; pres = []; recs = []
for ii in range(args.itr):
    # setting record of experiments
    setting = 'TranAP_{}_il{}_ol{}_ss{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_attn{}_itr{}'.format(args.data, 
                args.in_len, args.out_len, args.step_size, args.seg_len, args.win_size, args.factor,
                args.d_model, args.n_heads, args.e_layers, args.attn_ratio, ii)

    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    f1, pre, rec = exp.test(setting)
    f1s.append(f1)
    pres.append(pre)
    recs.append(rec)

print('*'*50)
print('Average ({} times):'.format(args.itr), 'F1', statistics.mean(f1s), 'Precision', statistics.mean(pres), 'Recall', statistics.mean(recs))
print('Sample Standard Deviation ({} times):'.format(args.itr), 'F1', statistics.stdev(f1s), 'Precision', statistics.stdev(pres), 'Recall', statistics.stdev(recs))
print('*'*50)

