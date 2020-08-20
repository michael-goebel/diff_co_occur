import torch, torchvision
import os, sys
from glob import glob
import numpy as np

sys.path.append('../utils/')
sys.path.append('utils/')


from pre_proc import CenterCrop, hwc2chw
from numpy_co_occur import CoOccur, co_occur_normalize
from numpy_dft import normed_dft
from imagenet_norm import imagenet_norm
from glob import glob
from data_gen import TrainDataGen, TestDataGen
from train import get_model, get_opt_and_loss, run_model, train_and_val, read_txt

def read_txt(fname):
	with open(fname) as f: f_list = f.read().split('\n')
	f_list = [f.replace('/ssd1/','/ssd2/') for f in f_list]
	return f_list




method = sys.argv[1]
model_name = sys.argv[2]
gpu = sys.argv[3]

os.environ['CUDA_VISIBLE_DEVICES'] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pre_proc_funcs = [CenterCrop(256), hwc2chw]


epochs = 5
bs_list = [16,16]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_dir = '/media/ssd2/mike/gan_data_trimmed/split_files/'

n_val = [1 for _ in range(16)]
n_val[0] = 4
n_val[1] = 2


tvt_list = ['train', 'val', 'test']
files_lists_dict = {tvt:[read_txt(f'{data_dir}reg_{tvt}_{t}.txt') for t in ['real','fake']] for tvt in tvt_list}
tr_gen = TrainDataGen(files_lists_dict['train'],[0,1],bs_list,pre_proc_funcs)
va_gen = TestDataGen(files_lists_dict['val'],[0,1],sum(bs_list),pre_proc_funcs)
te_gen = TestDataGen(files_lists_dict['test'],[0,1],sum(bs_list),pre_proc_funcs)


model = get_model(model_name,method).to(device)
model_parallel = model if len(gpu.split(','))==1 else torch.nn.DataParallel(model)

optimizer, loss_func = get_opt_and_loss(model_parallel)

out_dir = f'outputs/{method}_{model_name}/'

if not os.path.exists(out_dir): os.makedirs(out_dir)


train_and_val(out_dir,model_parallel,optimizer,loss_func,tr_gen,va_gen,n_val)











