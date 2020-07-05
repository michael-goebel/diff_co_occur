import sys, os, subprocess
from glob import glob

from PIL import Image
import numpy as np
import torch

sys.path.append('../utils/')
from my_alg import run_alg, savefig, co_occur


def get_losses(I1_orig,I1,I2):

	X_list = [I.detach() for I in [I1_orig, I1, I2]]
	C_list = [co_occur(X,ht_params) for X in X_list]

	img_rmse = torch.sqrt(torch.mean((X_list[0]-X_list[1])**2))
	cc_rmse = torch.sqrt(torch.mean(torch.stack([C_list[1][i]-C_list[2][i] for i in range(3)])**2))

	return img_rmse, cc_rmse

def print_err():
	img_rmse, cc_rmse = get_losses(I1_orig,I1,I2)
	print(f'Loss: {loss}\nImg RMSE: {img_rmse}\nCC RMSE: {cc_rmse}\n')



def round_rand(x): return torch.floor(x + torch.rand(x.shape,dtype=x.dtype,device=x.device))

ap_list = [{'n_steps': 300, 'sigma': 0.01, 'lamb': 0},
	   {'n_steps': 100, 'sigma': 0.01, 'lamb': 0},
	   {'n_steps': 100, 'sigma': 0.0, 'lamb': 0}]


#n_save = 10
#alg_params = {'n_steps': 300, 'sigma': 0.01, 'lamb': 0}
ht_params = {'n_layers': 9, 'n_bins': 256, 'interp': 'raised cos'}
optim_params = {'lr': 0.01, 'momentum': 0.9}

os.environ['CUDA_VISIBLE_DEVICES']='2'
real_list = glob('/home/mike/spring_2020/gan_images/real/*')
fake_list = glob('/home/mike/spring_2020/gan_images/fake/*')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

seed = int(sys.argv[1])
np.random.seed(seed)

real_img, fake_img = [np.random.choice(l) for l in [real_list,fake_list]]
print(real_img,'\n',fake_img)

I1_np, I2_np = [np.array(Image.open(f),dtype='single') for f in [fake_img,real_img]]
I1, I2 = [torch.tensor(i).to(device) for i in [I1_np,I2_np]]
I1_orig = I1.clone().detach()
I_list = [I1_orig,I1,I2]
loss = None

for alg_params in ap_list:

	print(alg_params)

	for i, loss in enumerate(run_alg(I1,I1_orig,I2,ht_params,optim_params,**alg_params)):
		if i % 100 == 0: print_err()
#			img_rmse, cc_rmse = get_losses(I1_orig,I1,I2)
#			print(f'Loss: {loss}\nImg RMSE: {img_rmse}\nCC RMSE: {cc_rmse}\n')
	print_err()
	I1.data = round_rand(I1)
	print('rounded')
	print_err()

#print_err()

#from hist import RaisedCos, hist_tree, hist_loss


#out_dir = f'outputs/seed_{seed}'
#if not os.path.exists(out_dir): os.makedirs(out_dir)
 
#for i, _ in enumerate(run_alg(I1,I1_orig,I2,ht_params,optim_params,**alg_params)):
#	if i % n_save == 0:
#		title = f'Step {i} of {alg_params["n_steps"]}'
#		savefig(*I_list,title,ht_params,os.path.join(out_dir,f'{i:03d}.png'))

#os.chdir(out_dir)
#subprocess.call(['convert', '-delay', '30', '*.png', 'out.mp4'])


