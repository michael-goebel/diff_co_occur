import sys, os
sys.path.append('../co_occur/utils/')
sys.path.append('../utils/')

from glob import glob
#from my_alg import savefig
from image_reader import image_reader, image_writer
import torch
from pre_proc import CenterCrop
import matplotlib.pyplot as plt

import itertools
import numpy as np

from my_alg import co_occur, get_losses

from matplotlib import cm

def cc2cmap(C):
	X = np.log(0.05 + C)
	X -= X.min()
	X /= X.max()
	X_out = cm.viridis(X)[:,:,:3]
	print(X_out.shape)
	return (255*X_out).astype('uint8')


def get_cc_mats(I1_orig,I1,I2,ht_params):

	col_labels = ['Source', 'Solution', 'Target']
	row_labels = ['Image', 'Red\nCo-Occur', 'Green\nCo-Occur', 'Blue\nCo-Occur']
	X_list = [I.detach() for I in [I1_orig, I1, I2]]
	C_list = [co_occur(X,ht_params) for X in X_list]

	return C_list


out_dir = 'block_diagram_imgs/'
if not os.path.exists(out_dir): os.mkdir(out_dir)


d_list = glob('/media/ssd2/mike/outputs_3/lambda_0.0/*/*/')


this_dir = d_list[21]

with open(this_dir + 'files.txt') as f: f_fake, f_real = f.read().split('\n')

f_adv = this_dir + 'output.png'


def file2torch(fname):
	img = image_reader(fname)
	img = CenterCrop(256)(img)
	return torch.Tensor(img)

f_list = [f.replace('ssd1','ssd2') for f in [f_fake,f_adv,f_real]]


I_list = [file2torch(f) for f in f_list]

ht_params = {'n_bins':256, 'interp': 'raised cos', 'n_layers':1}

C_list = get_cc_mats(*I_list,ht_params)


border = 6
delta = 30

L = 256
L_out = L + 2*(border+delta)

for j,label in enumerate(['gan', 'adv', 'real']):

	out_array = 255*np.ones((L_out,L_out,3)).astype('uint8')
	for i in range(3)[::-1]:
		d = i*delta
		out_array[d:d+L+2*border,d:d+L+2*border] = 0
		out_array[d+border:d+border+L,d+border:d+border+L] = cc2cmap(C_list[j][i][:L,:L])
	

	image_writer(out_dir + label + '_cc.png',out_array)

	img = np.zeros((L + 2*border,L + 2*border,3))
	img[border:L+border,border:L+border] = I_list[j]
	image_writer(out_dir + label + '_img.png', img)



#plt.imshow(out_array)
#plt.show()


#savefig(*I_list,ht_params,filename='good_output.png')



#print(f_real,f_fake,f_adv)




