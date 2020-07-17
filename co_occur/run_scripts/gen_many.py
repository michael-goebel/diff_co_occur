import sys, os, subprocess
from glob import glob

from PIL import Image
import numpy as np
import torch
import cv2

sys.path.append('../utils/')
from my_alg import run_alg, savefig, co_occur
import random
random.seed(234)


from tqdm import trange

def get_losses(I1_orig,I1,I2):

	X_list = [I.detach() for I in [I1_orig, I1, I2]]
	C_list = [co_occur(X,ht_params) for X in X_list]

	img_rmse = torch.sqrt(torch.mean((X_list[0]-X_list[1])**2))
	cc_rmse = torch.sqrt(torch.mean(torch.stack([C_list[1][i]-C_list[2][i] for i in range(3)])**2))

	return img_rmse, cc_rmse

def print_err(verbose=True):
	img_rmse, cc_rmse = get_losses(I1_orig,I1,I2)
	if verbose: print(f'Loss: {loss}\nImg RMSE: {img_rmse}\nCC RMSE: {cc_rmse}\n')
	return img_rmse, cc_rmse


def round_rand(x): return torch.floor(x + torch.rand(x.shape,dtype=x.dtype,device=x.device))


def center_crop(X):
	L = 256
	h,w,_ = X.shape
	ih = max(0,(h-L)//2)
	iw = max(0,(w-L)//2)
	return X[ih:,iw:][:L,:L]




ap_list = [{'n_steps': 300, 'sigma': 0.1, 'lamb': 0},
	   {'n_steps': 300, 'sigma': 0.01, 'lamb': 0},
	   {'n_steps': 100, 'sigma': 0.0, 'lamb': 0}]




ht_params = {'n_layers': 9, 'n_bins': 256, 'interp': 'raised cos'}
optim_params = {'lr': 0.01, 'momentum': 0.9}

os.environ['CUDA_VISIBLE_DEVICES']='2'

real_file = '/home/mike/spring_2020/prelim_cc_tests/train_val_splits/val_real.txt'
fake_file = '/home/mike/spring_2020/prelim_cc_tests/train_val_splits/val_fake.txt'

#fake_file = '/home/mike/spring_2020/prelim_cc_tests/train_val_splits/val_real.txt'
#real_file = '/home/mike/spring_2020/prelim_cc_tests/train_val_splits/val_fake.txt'


with open(real_file) as f: real_list = f.read().split('\n')
with open(fake_file) as f: fake_list = f.read().split('\n')

#real_list = glob('/home/mike/spring_2020/gan_images/real/*')
#fake_list = glob('/home/mike/spring_2020/gan_images/fake/*')

random.shuffle(real_list)
random.shuffle(fake_list)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


N = 50

out_dir = 'mod_outputs5/'
if not os.path.exists(out_dir+'imgs/'): os.makedirs(out_dir+'imgs/')

info_list = list()
verbose = False


for j in trange(N):

	real_img = real_list[j]
	fake_img = fake_list[j]
	#print(real_img,fake_img)

	I1_np, I2_np = [np.array(Image.open(f),dtype='single') for f in [fake_img,real_img]]

	I2_np = center_crop(I2_np)

	#I2_np = cv2.resize(I2_np,(256,256))
	#I1_np = cv2.resize(I1_np,(256,256))

	I1, I2 = [torch.tensor(i).to(device) for i in [I1_np,I2_np]]
	I1_orig = I1.clone().detach()
	I_list = [I1_orig,I1,I2]
	loss = None


	for alg_params in ap_list:

		for i, loss in enumerate(run_alg(I1,I1_orig,I2,ht_params,optim_params,**alg_params,verbose=verbose)):
			if i % 50 == 0: img_rmse, cc_rmse = print_err(verbose)
		print_err(verbose)
		I1.data = round_rand(I1)
		img_rmse, cc_rmse = print_err(verbose)

	savefig(I1_orig,I1,I2,f'{i}',ht_params,f'{out_dir}imgs/fig_{j}.png')

	X_out = I1.detach().cpu().numpy()
	Image.fromarray(X_out.astype('uint8')).save(f'{out_dir}imgs/{j}.png')
	info_list.append(f'{img_rmse},{cc_rmse},{real_img},{fake_img}')

with open(out_dir+'metadata.txt','w') as f: f.write('\n'.join(info_list))


