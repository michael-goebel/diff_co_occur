import sys
sys.path.append('../utils/')
#sys.path.append('../co_occur/utils/')

from glob import glob
from image_reader import image_reader
import torch
from pre_proc import CenterCrop, hwc2chw, chw2hwc
import matplotlib.pyplot as plt

from matplotlib import cm

import itertools
import numpy as np


def file2torch(fname):
	img = image_reader(fname)
	img = CenterCrop(256)(img)
	return torch.Tensor(img).float()/255

def get_imgs(dname):
	f_adv = dname + 'output.png'
	with open(dname + 'input_file.txt') as f: f_real = f.read().replace('ssd1','ssd2')
	X_list = [file2torch(f) for f in [f_real, f_adv]]
	X_list.append( 127*(X_list[1] - X_list[0]) + 0.5  )
	return X_list

attack_type = sys.argv[1]
attack_str = {'co_occur':'Co-Occur', 'dft':'DFT', 'direct':'Direct'}[attack_type]

d_list = glob(f'/home/mgoebel/summer_2020/diff_co_occur/pgd_attack/outputs_3/{attack_type}_resnet18_pretrained/*/*/')


np.random.seed(int(sys.argv[2]))

N = 8

inds = np.random.permutation(len(d_list))[:2*N]


img_list = [get_imgs(d_list[i]) for i in inds]

fig, axes = plt.subplots(N,6)

titles = ['GAN', 'PGD Adv', 'Diff']*2
for a,t in zip(axes[0],titles): a.set_title(t)
axes = np.vstack((axes[:,:3],axes[:,3:]))

for i in range(2*N):
	for j in range(3):
		axes[i,j].axis('off')
		axes[i,j].imshow(img_list[i][j])


fig.set_size_inches(10,12.5)
fig.suptitle(f'{attack_str} PGD Examples',fontsize=16)
fig.tight_layout()
line = plt.Line2D((0.5,0.5),(0,0.95),color='k',linewidth=2)
fig.add_artist(line)

fig.savefig(f'outputs/pgd_{attack_type}.png')


