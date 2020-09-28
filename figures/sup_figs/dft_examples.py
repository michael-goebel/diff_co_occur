import sys
#sys.path.append('../co_occur/utils/')
sys.path.append('../utils/')

sys.path.append('../detection/utils/')
from torch_pre_proc import DFTWithNorm

from glob import glob
from image_reader import image_reader
import torch
from pre_proc import CenterCrop, hwc2chw, chw2hwc
import matplotlib.pyplot as plt

from matplotlib import cm

import itertools
import numpy as np

from my_alg import co_occur, get_losses

def file2torch(fname):
	img = image_reader(fname)
	img = hwc2chw(CenterCrop(256)(img))
	return torch.Tensor(img)


def get_imgs(dname):
	
	f_adv = dname + 'output.png'
	with open(dname + 'files.txt') as f: f_fake, f_real = f.read().replace('ssd1','ssd2').split('\n')

	X_list = [file2torch(f) for f in [f_fake,f_adv,f_real]]
	
	F_list = [cm.viridis((DFTWithNorm()(X)[0]+1)/2) for X in X_list]
	F_list = [np.fft.fftshift(F,axes=(0,1)) for F in F_list]
	X_list = [X/255 for X in X_list]
	X_out = [chw2hwc(X.numpy()) for X in X_list]

	return X_out + F_list


lamb = sys.argv[1]

d_list = glob(f'/home/mgoebel/summer_2020/diff_co_occur/dft_attack/outputs_6/*/lambda_{lamb}/*/')

np.random.seed(int(sys.argv[2]))

N = 8

inds = np.random.permutation(len(d_list))[:N]

ht_params = {'n_bins':256, 'interp': 'raised cos', 'n_layers':1}

img_list = [get_imgs(d_list[i]) for i in inds]


fig, axes = plt.subplots(N,6)

titles = [t + ' ' + i for i in ['Image', 'DFT'] for t in ['GAN', 'Adv', 'Real']]

for a,t in zip(axes[0],titles): a.set_title(t)

for i in range(N):
	for j in range(6):
		axes[i,j].axis('off')
		axes[i,j].imshow(img_list[i][j])


fig.set_size_inches(10,12.5)
fig.suptitle('Gray-Box DFT Attack, ' + r'$\lambda = ' + lamb + r'$', fontsize=16)
fig.tight_layout()
fig.savefig(f'outputs/dft_{lamb}.png')




