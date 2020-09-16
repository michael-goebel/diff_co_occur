import torch, torchvision
import os, sys
from glob import glob
import numpy as np

sys.path.append('../utils/')
sys.path.append('utils/')


from pre_proc import CenterCrop, hwc2chw, JPEGFilter
from numpy_co_occur import CoOccur, co_occur_normalize
from numpy_dft import normed_dft
from imagenet_norm import imagenet_norm
from glob import glob
from data_gen import TrainDataGen, TestDataGen
from train import get_model, get_opt_and_loss, run_model, train_and_val, read_txt
from file_fetcher import unpack_names, get_files

def read_txt(fname):
	with open(fname) as f: f_list = f.read().split('\n')
	f_list = [f.replace('/ssd1/','/ssd2/') for f in f_list]
	return f_list


ind_start = int(sys.argv[1])
ind_end = int(sys.argv[2])
gpu = sys.argv[3]


os.environ['CUDA_VISIBLE_DEVICES']=gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pre_proc_funcs = [CenterCrop(256), JPEGFilter(75), hwc2chw]

bs_list = [16,16]


all_dirs = glob('outputs_jpeg/*/')


#good_dirs_trim = [d.split('/')[1] for d in good_dirs]

all_files = get_files('all_adv','test')
file_groups = ['real', 'GAN'] + unpack_names('all_adv')

file_labels = [0,] + [1,]*10
print(all_dirs)

for i, in_dir in enumerate(all_dirs[ind_start:ind_end]):

	#if os.path.exists(in_dir + 'test_results.txt'): print('skipping', in_dir); continue
	print(in_dir)
	meta_string = in_dir.split('/')[1]

	method = [m for m in ['co_occur','dft','direct'] if meta_string.startswith(m)][0]
	model_name = [m for m in ['resnet18','mobilenet'] if m in meta_string][0]
	print(in_dir, method, model_name)

	model = get_model(model_name,method).to(device)
	model.load_state_dict(torch.load(os.path.join(in_dir,'model.h5')))
	optimizer, loss_func = get_opt_and_loss(model)

	te_gen = TestDataGen(all_files, file_labels, sum(bs_list), pre_proc_funcs)

	loss, acc, conf = run_model(model, optimizer, loss_func, te_gen, False)
	print(conf.shape)

	out_str = ','.join(file_groups) + '\n' + ','.join([str(i) for i in conf[1]])
	with open(in_dir + 'test_results.txt', 'w+') as f: f.write(out_str)



