import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.co_occur.hist import hist_tree, hist_loss 

os.environ['CUDA_VISIBLE_DEVICES'] = str()

X1 = torch.Tensor([1,2,3]).view(-1,1).double()
X2 = torch.Tensor([2,3,4]).view(-1,1).double()

X1.requires_grad = True
torch.manual_seed(1234)

n_bins = 8
interp = 'raised cos'
n_steps = 1000
optim_params = {'lr': 0.001, 'momentum': 0.9}


fig, axes = plt.subplots(2,2)

for n_layers, ax_row in zip([1,3],axes):
	for sigma, ax in zip([0, 0.01],ax_row):

		X1 = torch.Tensor([1,2,3]).view(-1,1).double()
		X2 = torch.Tensor([2,3,4]).view(-1,1).double()

		X1.requires_grad = True

		H2 = hist_tree(X2,n_bins,n_layers,interp)
		optimizer = torch.optim.SGD((X1,),**optim_params)
		X_list = list()

		for i in range(n_steps):

			optimizer.zero_grad()

			X1.data = X1.data + sigma*torch.randn(X1.shape).double()
			X1_clamp = torch.clamp(X1,0,n_bins-1)
			X_list.append(X1_clamp.detach().numpy())
			H1 = hist_tree(X1_clamp,n_bins,n_layers,interp)

			loss = hist_loss(H1,H2)
			loss.backward()
			optimizer.step()

		X_arr = np.array(X_list).squeeze(2).T
		ax.set_ylim(0.75,4.25)
		for X_i in X_arr: ax.plot(X_i)
		title = 'L1' if n_layers == 1 else 'L1 Pyramid'
		title += r', $\sigma$ = ' + f'{sigma:0.2f}'
		ax.plot([0,0,0],[1,2,3],'k+')
		ax.plot([n_steps-1,]*3,[2,3,4],'ko',fillstyle='none')
		ax.plot([n_steps-1,]*3,X_arr[:,-1],'kx')

		ax.set_title(title)

		ax.set_xlabel('step number')
		ax.set_ylabel('value')


fig.set_size_inches(6,4)
fig.tight_layout()
plt.show()


