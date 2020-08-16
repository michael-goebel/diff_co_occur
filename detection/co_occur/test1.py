import torch, torchvision
import os, sys
from glob import glob


sys.path.append('../../utils/')
sys.path.append('../utils/')


from pre_proc import CenterCrop, hwc2chw
from numpy_co_occur import CoOccur, co_occur_normalize
from glob import glob
from data_gen import TrainDataGen, TestDataGen
from train import get_model, get_opt_and_loss, run_model, read_txt



pre_proc_funcs = [CenterCrop(256),
                 hwc2chw,
                 CoOccur(pairs='v',chan_first=True,L=256),
		 co_occur_normalize
                ]


epochs = 5
bs_list = [16,16]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_name = 'test_model1.pt'
data_dir = '/media/ssd1/mike/gan_data_trimmed/split_files/'

tvt_list = ['train', 'val', 'test']
files_lists_dict = {tvt:[read_txt(f'{data_dir}reg_{tvt}_{t}.txt') for t in ['real','fake']] for tvt in tvt_list}
tr_gen = TrainDataGen(files_lists_dict['train'],[0,1],bs_list,None,pre_proc_funcs)
va_gen = TestDataGen(files_lists_dict['val'],[0,1],sum(bs_list),pre_proc_funcs)
te_gen = TestDataGen(files_lists_dict['test'],[0,1],sum(bs_list),pre_proc_funcs)


model = get_model('resnet50')
optimizer, loss_func = get_opt_and_loss(model)


for i in range(epochs):

	print('Training')
	loss, acc, conf = run_model(model,optimizer,loss_func,tr_gen,True)
	print(loss,acc,conf)

	print('\nValidation')
	loss, acc, conf = run_model(model,optimizer,loss_func,va_gen,False)
	print(loss,acc,conf)


