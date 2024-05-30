import argparse
import os
import torch
import pickle

from exp.exp_TranAP import Exp_TranAP
from utilities.tools import load_args

parser = argparse.ArgumentParser(description='TranAP')

parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/TranAP/', help='location of the trained model')
parser.add_argument('--setting_name', type=str, default='TranAP_PSM_il48_ol24_ss6_sl12_win2_fa10_dm256_nh4_el3_attn0.25_itr0/', help='name of the experiment')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

parser.add_argument('--data_split', type=str, default=[0.8, 0.2], help='data split of train, vali, test')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.use_multi_gpu = False

args.checkpoint_dir = os.path.join(args.checkpoint_root, args.setting_name)
hyper_parameters = load_args(os.path.join(args.checkpoint_dir, 'args.json'))

#load the pre-trained model
args.in_len = hyper_parameters['in_len']; args.out_len = hyper_parameters['out_len']; 
args.step_size = hyper_parameters['step_size']; args.seg_len = hyper_parameters['seg_len']; 
args.win_size = hyper_parameters['win_size']; args.factor = hyper_parameters['factor'];
args.data_dim = hyper_parameters['data_dim']; 
args.d_model = hyper_parameters['d_model']; args.d_ff = hyper_parameters['d_ff']; args.n_heads = hyper_parameters['n_heads'];
args.e_layers = hyper_parameters['e_layers']; args.dropout = hyper_parameters['dropout']; args.attn_ratio = hyper_parameters['attn_ratio'];
exp = Exp_TranAP(args)
print(args)
model_path = args.checkpoint_dir + 'checkpoint.pth'
exp.model1.load_state_dict(torch.load(model_path)['model1'])
exp.model2.load_state_dict(torch.load(model_path)['model2'])

#load the data
args.root_path = hyper_parameters['root_path']; args.data = hyper_parameters['data'];

f1, precision, recall, mae, mse = exp.eval(args.setting_name)

folder_path = './results/TranAP/' + args.setting_name
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
log_file = open(folder_path+'metric.log', 'w')
log_file.write('Data Split: {}\n'.format(args.data_split))
log_file.write('Input Length:{}   Output Length:{}\n'.format(args.in_len, args.out_len))
log_file.write("F1:{}\nPRE:{}\nREC:{}\n".format(f1, precision, recall))
log_file.write('MAE:{}\nMSE:{}\n'.format(mae, mse))
log_file.close()
