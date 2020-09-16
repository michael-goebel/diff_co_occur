import torch
import os, sys
sys.path.append('../utils/')
sys.path.append('utils/')

from pre_proc import CenterCrop, hwc2chw, JPEGFilter
from data_gen import TrainDataGen, TestDataGen
from train import get_model, get_opt_and_loss, run_model, train_and_val, read_txt
from file_fetcher import get_files


method = sys.argv[1]
model_name = sys.argv[2]
data_group = sys.argv[3]
gpu = sys.argv[4]

pretrained = True

os.environ['CUDA_VISIBLE_DEVICES'] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pre_proc_funcs = [CenterCrop(256), JPEGFilter(75), hwc2chw]

bs_real = 20 if data_group == 'all_adv' else 16

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_dir = '/media/ssd2/mike/gan_data_trimmed/split_files/'

n_val = [1 for _ in range(16)]
n_val[0] = 4
n_val[1] = 2


files_lists_tr = get_files(data_group,'train')
files_lists_va = get_files(data_group,'val')

n_fake = len(files_lists_tr) - 1
l_labels = [0,] + [1,]*n_fake
bs_list = [bs_real,] + [bs_real//n_fake]*n_fake


tr_gen = TrainDataGen(files_lists_tr,l_labels,bs_list,pre_proc_funcs)
va_gen = TestDataGen(files_lists_va,l_labels,sum(bs_list),pre_proc_funcs)

model = get_model(model_name,method,pretrained).to(device)
model_parallel = model if len(gpu.split(','))==1 else torch.nn.DataParallel(model)
optimizer, loss_func = get_opt_and_loss(model_parallel)

pt = 'pretrained' if pretrained else 'randinit'
out_dir = f'outputs_jpeg/{method}_{model_name}_{data_group}/'
print(out_dir)

if not os.path.exists(out_dir): os.makedirs(out_dir)

train_and_val(out_dir,model,model_parallel,optimizer,loss_func,tr_gen,va_gen,n_val,show_pbar=True)



