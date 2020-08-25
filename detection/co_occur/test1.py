import torch, torchvision
import os, sys
from glob import glob
import numpy as np

sys.path.append('../../utils/')
sys.path.append('../utils/')


from pre_proc import CenterCrop, hwc2chw
from numpy_co_occur import CoOccur, co_occur_normalize
from glob import glob
from data_gen import TrainDataGen, TestDataGen
from train import get_model, get_opt_and_loss, run_model, train_and_val, read_txt

def read_txt(fname):
	with open(fname) as f: f_list = f.read().split('\n')
	f_list = [f.replace('/ssd1/','/ssd2/') for f in f_list]
	return f_list

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


device = 'cuda' if torch.cuda.is_available() else 'cpu'



pre_proc_funcs = [CenterCrop(256),
                 hwc2chw,
                 CoOccur(pairs='v',chan_first=True,L=256),
		 co_occur_normalize
                ]


epochs = 5
bs_list = [16,16]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_name = 'test_model1.pt'
data_dir = '/media/ssd2/mike/gan_data_trimmed/split_files/'

n_val = [4,2,1,1,1]


tvt_list = ['train', 'val', 'test']
files_lists_dict = {tvt:[read_txt(f'{data_dir}reg_{tvt}_{t}.txt') for t in ['real','fake']] for tvt in tvt_list}
tr_gen = TrainDataGen(files_lists_dict['train'],[0,1],bs_list,pre_proc_funcs)
va_gen = TestDataGen(files_lists_dict['val'],[0,1],sum(bs_list),pre_proc_funcs)
te_gen = TestDataGen(files_lists_dict['test'],[0,1],sum(bs_list),pre_proc_funcs)

model_name = 'resnet18'

model = get_model(model_name).to(device)
optimizer, loss_func = get_opt_and_loss(model)


#for i,nv in enumerate(n_val):

out_dir = f'outputs_{model_name}/'

if not os.path.exists(out_dir): os.mkdir(out_dir)


train_and_val(out_dir,model,optimizer,loss_func,tr_gen,va_gen,n_val)






#	tr_gen.shuffle()

#	tgd_inds = np.linspace(0,tr_dg.L,nv+1)
#	print(tgd_inds)



#	for j in range(nv):



#	print('Training')
#	loss, acc, conf = run_model(model,optimizer,loss_func,tr_gen,True,ind_start=3,n_batch=4)
#	print(loss,acc,conf)

#	print('\nValidation')
#	loss, acc, conf = run_model(model,optimizer,loss_func,va_gen,False)
#	print(loss,acc,conf)








