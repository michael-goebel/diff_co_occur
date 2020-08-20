import torch
import os, sys
sys.path.append('../utils/')
sys.path.append('utils/')

from pre_proc import CenterCrop, hwc2chw
from data_gen import TrainDataGen, TestDataGen
from train import get_model, get_opt_and_loss, run_model, train_and_val, read_txt

def read_txt(fname):
	with open(fname) as f: f_list = f.read().split('\n')
	f_list = [f.replace('/ssd1/','/ssd2/') for f in f_list]
	return f_list


method = sys.argv[1]
model_name = sys.argv[2]
pretrained = (sys.argv[3] == 'True')
gpu = sys.argv[4]

os.environ['CUDA_VISIBLE_DEVICES'] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pre_proc_funcs = [CenterCrop(256), hwc2chw]

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

model = get_model(model_name,method,pretrained).to(device)
model_parallel = model if len(gpu.split(','))==1 else torch.nn.DataParallel(model)
optimizer, loss_func = get_opt_and_loss(model_parallel)

pt = 'pretrained' if pretrained else 'randinit'
out_dir = f'outputs_3/{method}_{model_name}_{pt}/'
print(out_dir)

if not os.path.exists(out_dir): os.makedirs(out_dir)

train_and_val(out_dir,model,model_parallel,optimizer,loss_func,tr_gen,va_gen,n_val,show_pbar=True)



