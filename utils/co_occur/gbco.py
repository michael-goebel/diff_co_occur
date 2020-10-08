import os, glob, itertools, subprocess

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.co_occur.hist import hist_tree, hist_loss


def img2pairs(X,ch_first=False):
	if ch_first: return [torch.stack((X[i,:-1].reshape(-1),X[i,1:].reshape(-1)),dim=1) for i in range(X.shape[0])]
	else: return [torch.stack((X[:-1,:,i].reshape(-1),X[1:,:,i].reshape(-1)),dim=1) for i in range(X.shape[-1])]	


rgb_pairs = [[0,1],[0,2],[1,2]]
def img2pairs_cband(X,ch_first=False):
	if ch_first:
		out = [torch.stack((X[i,:-1,:-1].reshape(-1),X[i,1:,1:].reshape(-1)),dim=1) for i in range(3)]
		out += [torch.stack((X[i].reshape(-1),X[j].reshape(-1)),dim=1) for i,j in rgb_pairs]
		return out
	else:
		out = [torch.stack((X[:-1,:-1,i].reshape(-1),X[1:,1:,i].reshape(-1)),dim=1) for i in range(3)]
		out += [torch.stack((X[:,:,i].reshape(-1),X[:,:,j].reshape(-1)),dim=1) for i,j in rgb_pairs]
		return out
	



def co_occur(X,ht_params): return [hist_tree(X_i,ht_params['n_bins'],1,ht_params['interp'])[0] for X_i in img2pairs(X)]


def round_rand(x): return torch.floor(x + torch.rand(x.shape,dtype=x.dtype,device=x.device))


def run_alg(I1_orig,I1,I2,ht_params,optim_params,n_steps,sigma,lamb,cband=False):
	
	pairs_fxn = img2pairs_cband if cband else img2pairs
	I1.requires_grad = True
	optimizer = torch.optim.SGD((I1,),**optim_params)
	H2 = [hist_tree(X2_i,**ht_params) for X2_i in pairs_fxn(I2)]

	for i in range(n_steps):
		optimizer.zero_grad()
		noise = sigma*torch.randn(I1.shape).type(I1.dtype).to(I1.device)
		I1.data = torch.clamp(I1+noise,0,ht_params['n_bins']-1)
		H1 = [hist_tree(X1_i,**ht_params) for X1_i in pairs_fxn(I1)]
		loss = sum([hist_loss(H1_i,H2_i) for H1_i, H2_i in zip(H1,H2)]) + lamb*torch.sum(torch.abs(I1_orig-I1))
		loss.backward()
		optimizer.step()
		I1.data = torch.clamp(I1,0,ht_params['n_bins']-1)
		yield loss.detach()


def get_losses(I1_orig,I1,I2,ht_params):

	X_list = [I.detach() for I in [I1_orig, I1, I2]]
	C_list = [co_occur(X,ht_params) for X in X_list]
	img_l1 = torch.mean(torch.abs(X_list[0]-X_list[1]))
	cc_l1 = torch.mean(torch.abs(torch.stack([C_list[1][i]-C_list[2][i] for i in range(3)])))
	return img_l1, cc_l1


def savefig(I1_orig,I1,I2,ht_params,filename='fig.png',title=str()):

	col_labels = ['Source', 'Solution', 'Target']
	row_labels = ['Image', 'Red\nCo-Occur', 'Green\nCo-Occur', 'Blue\nCo-Occur']
	X_list = [I.detach() for I in [I1_orig, I1, I2]]
	C_list = [co_occur(X,ht_params) for X in X_list]
	img_l1, cc_l1 = get_losses(*X_list,ht_params)

	fig,axes = plt.subplots(4,3)
	for a, X in zip(axes[0],X_list): a.imshow(X.cpu().numpy()/255)
	for a, C in zip(axes[1:].T.reshape(-1),itertools.chain(*C_list)): a.imshow(np.log(1+C.cpu().numpy()))
	for a, l in zip(axes[0],col_labels): a.set_title(l)
	for a, l in zip(axes[:,0],row_labels): a.set_ylabel(l)
	for a in axes.reshape(-1): a.set_xticks([]); a.set_yticks([])

	fig.suptitle(title)
	fig.text(0.02,0.02,f'Image L1 Loss: {img_l1:0.3f}\nCo-Occur L1 Loss: {cc_l1:0.3f}',fontsize=14)
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

		all_figs = os.path.join(self.base_dir,'*.png')
		vid_file = os.path.join(self.base_dir,'output_fig.mp4')
		subprocess.call(['convert', '-delay', str(self.delay), all_figs, vid_file])
		if self.remove_figs:
			for f in glob.glob(os.path.join(self.base_dir,'*.png')):
				os.remove(f)
	

