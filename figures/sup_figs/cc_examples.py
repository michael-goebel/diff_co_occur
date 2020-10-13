from glob import glob
import torch
import matplotlib.pyplot as plt
import itertools
import numpy as np

from utils.image.image_io import image_reader
from utils.pre_proc import CenterCrop
from utils.co_occur.gbco import co_occur, get_losses


def act(x): return np.sign(x)*np.log(0.05+np.abs(x))

def norm_cc(X):
	X_out = act(X)
	X_out /= X_out.max()
	return X_out


def savefig(I1_orig,I1,I2,ht_params,filename='fig.png',title=str()):
	col_labels = ['Source', 'Solution', 'Target']
	row_labels = ['Image', 'Red\nCo-Occur', 'Green\nCo-Occur', 'Blue\nCo-Occur']
	X_list = [I.detach() for I in [I1_orig, I1, I2]]
	C_list = [co_occur(X,ht_params) for X in X_list]
	delta_C = [C_list[i][1] - C_list[i][2] for i in range(3)]
	fig, axes = plt.subplots(1,3)
	for i in range(3): axes[i].imshow(act(delta_C[i]))
	plt.savefig('foo.png')
	plt.close()


def file2torch(fname):
	img = image_reader(fname)
	img = CenterCrop(256)(img)
	return torch.Tensor(img)


def get_imgs(dname):
	f_adv = dname + 'output.png'
	with open(dname + 'files.txt') as f: f_fake, f_real = f.read().replace('ssd1','ssd2').split('\n')
	X_list = [file2torch(f) for f in [f_fake,f_adv,f_real]]
	C_list = [co_occur(X,ht_params)[0] for X in X_list]
	X_list = [X/255 for X in X_list]
	return [X.numpy() for X in X_list + C_list]




lamb = sys.argv[1]

d_list = glob(f'/media/ssd2/mike/outputs_3/lambda_{lamb}/*/*/')

np.random.seed(int(sys.argv[2]))

N = 8

inds = np.random.permutation(len(d_list))[:N]

ht_params = {'n_bins':256, 'interp': 'raised cos', 'n_layers':1}

img_list = [get_imgs(d_list[i]) for i in inds]


fig, axes = plt.subplots(N,6)

titles = [t + ' ' + i for i in ['Image', 'Co-Occur'] for t in ['GAN', 'Adv', 'Real']]

for a,t in zip(axes[0],titles): a.set_title(t)

for i in range(N):
	for j in range(6):
		axes[i,j].axis('off')
		axes[i,j].imshow(img_list[i][j])


fig.set_size_inches(10,12.5)
fig.suptitle('Gray-Box Co-Occur Attack, ' + r'$\lambda = ' + lamb + r'$', fontsize=16)
plt.tight_layout()
fig.savefig(f'outputs/co_occur_{lamb}.png')




