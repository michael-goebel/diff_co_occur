import sys
sys.path.append('../co_occur/utils/')
sys.path.append('../utils/')

from glob import glob
#from my_alg import savefig
from image_reader import image_reader
import torch
from pre_proc import CenterCrop
import matplotlib.pyplot as plt

import itertools
import numpy as np

from my_alg import co_occur, get_losses

def act(x): return np.sign(x)*np.log(1+np.abs(x))


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

#	plt.show()		


#	 img_l1, cc_l1 = get_losses(*X_list,ht_params)

#	 fig,axes = plt.subplots(4,3)
#	 for a, X in zip(axes[0],X_list): a.imshow(X.cpu().numpy()/255)
#	 for a, C in zip(axes[1:].T.reshape(-1),itertools.chain(*C_list)): a.imshow(np.log(0.05+C.cpu().numpy()))
#	 for a, l in zip(axes[0],col_labels): a.set_title(l)
#	 for a, l in zip(axes[:,0],row_labels): a.set_ylabel(l)
#	 for a in axes.reshape(-1): a.set_xticks([]); a.set_yticks([])

#	 fig.suptitle(title)
#	 fig.text(0.02,0.02,f'Image L1 Loss: {img_l1:0.3f}\nCo-Occur L1 Loss: {cc_l1:0.3f}',fontsize=12)
#	 fig.set_size_inches(6,6)
#	 fig.tight_layout(rect=[0,0.067,1,1])
#	 fig.savefig(filename)
#	 plt.close()





d_list = glob('/media/ssd2/mike/outputs_3/lambda_0.0/*/*/')


this_dir = d_list[16]

with open(this_dir + 'files.txt') as f: f_fake, f_real = f.read().split('\n')

f_adv = this_dir + 'output.png'


def file2torch(fname):
	img = image_reader(fname)
	img = CenterCrop(256)(img)
	return torch.Tensor(img)

f_list = [f.replace('ssd1','ssd2') for f in [f_fake,f_adv,f_real]]


I_list = [file2torch(f) for f in f_list]

ht_params = {'n_bins':256, 'interp': 'raised cos', 'n_layers':1}

savefig(*I_list,ht_params,filename='good_output.png')



#print(f_real,f_fake,f_adv)




