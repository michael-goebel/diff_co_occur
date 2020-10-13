import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.co_occur.hist import Hist, hist_tree


def hist_loss_L1(H1,H2): return sum([(2**i)*torch.sum(torch.abs(h1-h2)) for i,(h1,h2) in enumerate(zip(H1,H2))])
def hist_loss_L2(H1,H2): return sum([(2**i)*torch.sum((h1-h2)**2)**(1/2) for i,(h1,h2) in enumerate(zip(H1,H2))])


os.environ['CUDA_VISIBLE_DEVICES'] = str()

X1 = torch.tensor([1,2,3]).view(-1,1).double()
X2 = torch.tensor([2,3,4]).view(-1,1).double()
X1.requires_grad = True

params_list = [[hist_loss_L1, 1, 'L1'], [hist_loss_L2, 1, 'L2'], [hist_loss_L1, 3, 'L1 Pyramid']]

n_bins = 8
interp = 'raised cos'

v_min = 1
v_max = 4

N = X1.shape[0]
M = 100
x = np.linspace(v_min,v_max,M)

fig, axes = plt.subplots(N,len(params_list))

for (loss_func, n_layers, name), ax in zip(params_list,axes.T):

	X_list = list()
	loss_arr = np.empty((N,M))
	H2 = hist_tree(X2,n_bins,n_layers,interp)

	for ind in range(X1.shape[0]):
		X0 = X1.clone()
		for i,xi in enumerate(x):
			X0[ind] = xi
			H0 = hist_tree(X0,n_bins,n_layers,interp)
			loss_arr[ind,i] = loss_func(H0,H2)

	H1 = hist_tree(X1,n_bins,n_layers,interp)
	loss = loss_func(H1,H2).detach()
	y_max = loss_arr.max()
	ax[0].set_title(name)

	for i in range(N):
		ax[i].plot(x,loss_arr[i])
		ax[i].set_ylim(-0.5,y_max+0.5)
		ax[i].plot(X1[i].detach(),loss,'o', color='#1f77b4')

		xmin,xmax = ax[i].get_xlim()
		ax[i].plot([xmin,xmax],[0,0],':',color='#808080')
		ax[i].set_xlim(xmin,xmax)


for i in range(N): axes[i,0].set_ylabel(f'Loss for $x_{i+1}$')
fig.set_size_inches(6,5)
fig.tight_layout()

plt.show()


