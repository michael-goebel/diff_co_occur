import torch
import itertools
import numpy as np
import os
import subprocess
from hist import hist_tree, hist_loss
from tqdm import trange
import matplotlib.pyplot as plt
from glob import glob

def img2pairs(X): return [torch.stack((X[:-1,:,i].view(-1),X[1:,:,i].view(-1)),dim=1) for i in range(X.shape[-1])]


def co_occur(X,ht_params): return [hist_tree(X_i,ht_params['n_bins'],1,ht_params['interp'])[0] for X_i in img2pairs(X)]


def round_rand(x): return torch.floor(x + torch.rand(x.shape,dtype=x.dtype,device=x.device))


def get_losses(I1_orig,I1,I2):

        X_list = [I.detach() for I in [I1_orig, I1, I2]]
        C_list = [co_occur(X,ht_params) for X in X_list]
        img_rmse = torch.sqrt(torch.mean((X_list[0]-X_list[1])**2))
        cc_rmse = torch.sqrt(torch.mean(torch.stack([C_list[1][i]-C_list[2][i] for i in range(3)])**2))
        return img_rmse, cc_rmse


def run_alg(I1_orig,I1,I2,ht_params,optim_params,n_steps,sigma,lamb,verbose=False):

	I1.requires_grad = True
	optimizer = torch.optim.SGD((I1,),**optim_params)
	H2 = [hist_tree(X2_i,**ht_params) for X2_i in img2pairs(I2)]
	my_range = trange if verbose else range

	for i in my_range(n_steps):
		optimizer.zero_grad()
		noise = sigma*torch.randn(I1.shape).type(I1.dtype).to(I1.device)
		I1.data = torch.clamp(I1+noise,0,ht_params['n_bins']-1)
		H1 = [hist_tree(X1_i,**ht_params) for X1_i in img2pairs(I1)]
		loss = sum([hist_loss(H1_i,H2_i) for H1_i, H2_i in zip(H1,H2)])
		loss.backward()
		optimizer.step()
		I1.data = torch.clamp(I1,0,ht_params['n_bins']-1)
		yield loss.detach()


def get_losses(I1_orig,I1,I2,ht_params):

	X_list = [I.detach() for I in [I1_orig, I1, I2]]
	C_list = [co_occur(X,ht_params) for X in X_list]
	img_rmse = torch.sqrt(torch.mean((X_list[0]-X_list[1])**2))
	cc_rmse = torch.sqrt(torch.mean(torch.stack([C_list[1][i]-C_list[2][i] for i in range(3)])**2))

	return img_rmse, cc_rmse



def savefig(I1_orig,I1,I2,ht_params,filename='fig.png',title=str()):

	col_labels = ['Source', 'Solution', 'Target']
	row_labels = ['Image', 'Red', 'Green', 'Blue']
	X_list = [I.detach() for I in [I1_orig, I1, I2]]
	C_list = [co_occur(X,ht_params) for X in X_list]
	img_rmse, cc_rmse = get_losses(*X_list,ht_params)

	fig,axes = plt.subplots(4,3)
	for a, X in zip(axes[0],X_list): a.imshow(X.cpu().numpy()/255)
	for a, C in zip(axes[1:].T.reshape(-1),itertools.chain(*C_list)): a.imshow(np.log(1+C.cpu().numpy()))
	for a, l in zip(axes[0],col_labels): a.set_title(l)
	for a, l in zip(axes[:,0],row_labels): a.set_ylabel(l)
	for a in axes.reshape(-1): a.set_xticks([]); a.set_yticks([])

	fig.suptitle(title)
	fig.text(0.02,0.02,f'Image RMSE: {img_rmse:0.3f}\nCo-Occur RMSE: {cc_rmse:0.3f}',fontsize=14)
	fig.set_size_inches(8,8)
	fig.savefig(filename)
	plt.close()


class VideoGen:

	def __init__(self,base_dir,ap_list,delay=30,remove_figs=True):

		self.base_dir = base_dir
		self.delay = delay
		self.remove_figs = remove_figs
		self.i = 0
		self.N = sum([ap['n_steps'] for ap in ap_list])
		if not os.path.exists(base_dir): os.makedirs(base_dir)

	def add_fig(self,I1_orig,I1,I2,ht_params,index):
		title = f'Step {index} of {self.N}'
		filename = os.path.join(self.base_dir,f'{self.i:05d}.png')
		savefig(I1_orig,I1,I2,ht_params,filename,title)
		self.i += 1

	def save(self):
		#os.chdir(self.base_dir)

		all_figs = os.path.join(self.base_dir,'*.png')
		vid_file = os.path.join(self.base_dir,'output.mp4')
		subprocess.call(['convert', '-delay', str(self.delay), all_figs, vid_file])
		if self.remove_figs:
			for f in glob(os.path.join(self.base_dir,'*.png')):
				os.remove(f)
	
		#subprocess.call(['convert', '-delay', str(self.delay), os.path.join(self.base_dir,'*.png'), os.path.join(self.base_dir,'output.mp4')])


		#if self.remove_figs: subprocess.call(['ls'])








