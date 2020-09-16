import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append('../utils/')
from hist import RaisedCos, L1Dist, Hist, hist_tree, hist_loss


os.environ['CUDA_VISIBLE_DEVICES'] = str()

# X1 is source, X2 is target. Demostrates problem of transporting from 0 to 6
X1 = torch.tensor([0,1,2]).view(-1,1).double()
X2 = torch.tensor([1,2,3]).view(-1,1).double()
X1.requires_grad = True


n_bins = 4
n_layers = 3
interp = 'raised cos'
n_steps = 200

H2 = hist_tree(X2,n_bins,n_layers,interp)
#optimizer = torch.optim.SGD((X1,),**optim_params)
X_list = list()

N = X1.shape[0]
M = 100


x = np.linspace(0,3,M)



loss_arr = np.empty((N,M))

for ind in range(X1.shape[0]):

	X0 = X1.clone()
	for i,xi in enumerate(x):
		X0[ind] = xi
		H0 = hist_tree(X0,n_bins,n_layers,interp)
		loss_arr[ind,i] = hist_loss(H0,H2)

H1 = hist_tree(X1,n_bins,n_layers,interp)
loss = hist_loss(H1,H2).detach()
		

y_max = loss_arr.max()

fig,axes = plt.subplots(N,1)

for i in range(N):
	axes[i].plot(x,loss_arr[i])
	axes[i].set_ylim(-0.5,y_max+0.5)
	axes[i].plot(X1[i].detach(),loss,'o', color='#1f77b4')
	#axes[i].plot([0,3],[0,0],':',color='#808080')
#	axes[i].plot([0,0,0,3,3,3],[y_max+0.5,-0.5,0,0,-0.5,y_max+0.5],':',color='#808080')

	xmin,xmax = axes[i].get_xlim()
	print(xmin,xmax)
	axes[i].plot([xmin,xmax],[0,0],':',color='#808080')
	axes[i].plot([0,0],[y_max+0.5,-0.5],':',color='#808080')
	axes[i].plot([3,3],[y_max+0.5,-0.5],':',color='#808080')
	axes[i].set_xlim(xmin,xmax)

plt.show()

	
#print(loss_arr)



#for i in range(n_steps):

#	optimizer.zero_grad()

#	X1.data = X1.data + sigma*torch.randn(X1.shape).double()
#	X1_clamp = torch.clamp(X1,0,n_bins-1)
#	X_list.append(X1_clamp.detach().numpy())
#	H1 = hist_tree(X1_clamp,n_bins,n_layers,interp)

#	loss = hist_loss(H1,H2)
#	loss.backward()
#	optimizer.step()

#X_arr = np.array(X_list).squeeze(2).T
#for X_i in X_arr: plt.plot(X_i)
#plt.show()


