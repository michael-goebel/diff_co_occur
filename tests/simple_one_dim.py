import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append('../utils/')
from hist import RaisedCos, L1Dist, Hist, hist_tree, hist_loss


os.environ['CUDA_VISIBLE_DEVICES'] = str()

# X1 is source, X2 is target. Demostrates problem of transporting from 0 to 6
X1 = torch.tensor([0,0,1,2,3,4,5,6]).view(-1,1).double()
X2 = torch.tensor([0,1,2,3,4,5,6,6]).view(-1,1).double()
X1.requires_grad = True


n_bins = 8
n_layers = 1
sigma = 0.001
interp = 'raised cos'
n_steps = 200
optim_params = {'lr': 0.001, 'momentum': 0.9}

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
for X_i in X_arr: plt.plot(X_i)
plt.show()


