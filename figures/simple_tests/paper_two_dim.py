import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../utils/')
from hist import RaisedCos, L1Dist, Hist, hist_tree, hist_loss
from scipy import ndimage

os.environ['CUDA_VISIBLE_DEVICES'] = str()

torch.manual_seed(128)

L = 6

#X1 = torch.randint(0,4,(L,2)).double()
#X2 = torch.randint(0,4,(L,2)).double()

#X1 = torch.Tensor([[3,3,1,2,1],\
#		   [1,1,0,3,2]]).double().T

#X2 = torch.Tensor([[2,2,0,3,2],\
#		   [0,1,0,3,2]]).double().T


X1 = torch.Tensor([[1,1,1,2,2,0],\
		   [1,1,2,2,3,3]]).double().T

X2 = torch.Tensor([[2,3,2,2,3,0],\
		   [1,2,3,0,3,2]]).double().T


X1_orig = X1.clone().detach()
X2_orig = X2.clone().detach()

#X1.requires_grad = True

markersize = 9

n_bins = 4
#n_layers = 1
#sigma = 0.001
interp = 'raised cos'
n_steps = 300
lamb = 0
optim_params = {'lr': 0.001, 'momentum': 0.9}
n_arrow = 30

fig, axes = plt.subplots(2,2)

#arrow_len = 0.5
arrow_width = 0.08

min_arrow_len = 0.01

gauss_sigma = 5


def my_plot(X,n_arrow,ax,ind):
	a_inds = (X.shape[1]*(np.arange(n_arrow)+1))//(n_arrow+1)
	a_inds += np.random.randint(-2,3,a_inds.shape)

#	print(a_inds)

	color = plt.rcParams['axes.prop_cycle'].by_key()['color'][ind]
	ax.plot(*X,color=color)
#	print(X)
#	color = list(plt.rcParams['axes.prop_cycle'].by_key())[ind]
	#color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
	print(color)
	X_b = ndimage.gaussian_filter1d(X,gauss_sigma)

	for a in a_inds:
		x,y = X[:,a]
		delta = X_b[:,a+1] - X_b[:,a]
		if np.linalg.norm(delta) > min_arrow_len:
			delta *= min_arrow_len/np.linalg.norm(delta)
	
			ax.arrow(x,y,*delta,head_width=arrow_width,edgecolor=color,facecolor=color)

#			delta *= arrow_len/np.linalg.norm(delta)
#		else: delta = np.zeros(2)
#		ax.arrow(x,y,*deltai,)


#		ax.arrow(X[1,a],X[0,a],X[1,a+1]-X[1,a],X[0,a+1]-X[0,a])



#	ax.arrow(X[1][a_inds],X[0][a_inds],X[1][a_inds+1]-X[1][a_inds],X[0][a_inds+1]-X[0][a_inds])
#



for n_layers,ax_row in zip([1,3],axes):
	for sigma,ax in zip([0,0.001],ax_row):

		#X1 = torch.randint(0,4,(L,2)).double()
		#X2 = torch.randint(0,4,(L,2)).double()

		#X1_orig = X1.clone().detach()
		X1 = X1_orig.clone()
		X2 = X2_orig.clone()


		X1.requires_grad = True



		H2 = hist_tree(X2,n_bins,n_layers,interp)
		optimizer = torch.optim.SGD((X1,),**optim_params)
		X_list = list()

		for i in range(n_steps):

			optimizer.zero_grad()
			X1.data = torch.clamp(X1+sigma*torch.randn(X1.shape).double(),0,n_bins-1)

			X_list.append(X1.detach().numpy())
			H1 = hist_tree(X1,n_bins,n_layers,interp)
			loss = hist_loss(H1,H2) + lamb*torch.sum((X1-X1_orig)**2)

			loss.backward()
			optimizer.step()

		X_arr = np.array(X_list).transpose(1,2,0)
#		for X_i in X_arr: ax.plot(*X_i)
		for i,X_i in enumerate(X_arr): my_plot(X_i,n_arrow,ax,i)

		X1_orig_np = X1_orig.detach().numpy().T
		X1_np = X1.detach().numpy().T
		X2_np = X2.detach().numpy().T

		if (n_layers == 1) & (sigma == 0.001):
			ax.plot(*X1_orig_np,'k+',label='Source',markersize=markersize)
			ax.plot(*X1_np,'kx',label='Output',markersize=markersize)
			ax.plot(*X2_np,'ko',fillstyle='none',label='Target',markersize=markersize)
			ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)		
	
		else:
			ax.plot(*X1_orig_np,'k+',markersize=markersize)
			ax.plot(*X1_np,'kx',markersize=markersize)
			ax.plot(*X2_np,'ko',fillstyle='none',markersize=markersize)



		title = 'L1' if n_layers==1 else 'L1 Pyramid'
		title += r', $\sigma$ = ' + f'{sigma:0.3f}'
		ax.set_title(title)

		ax.set_aspect(1)
		#ax.legend()

fig.set_size_inches(6,6)
plt.tight_layout()

plt.show()



