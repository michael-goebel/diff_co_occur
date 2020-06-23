import sys, os, subprocess
from glob import glob

from PIL import Image
import numpy as np
import torch

sys.path.append('../utils/')
from my_alg import run_alg, savefig


n_save = 10
alg_params = {'n_steps': 300, 'sigma': 0.01, 'lamb': 0}
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

out_dir = f'outputs/seed_{seed}'
if not os.path.exists(out_dir): os.makedirs(out_dir)
 
for i, _ in enumerate(run_alg(I1,I1_orig,I2,ht_params,optim_params,**alg_params)):
	if i % n_save == 0:
		title = f'Step {i} of {alg_params["n_steps"]}'
		savefig(*I_list,title,ht_params,os.path.join(out_dir,f'{i:03d}.png'))

os.chdir(out_dir)
subprocess.call(['convert', '-delay', '30', '*.png', 'out.mp4'])


