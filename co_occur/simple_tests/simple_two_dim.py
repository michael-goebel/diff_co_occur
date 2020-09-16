import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../utils/')
from hist import RaisedCos, L1Dist, Hist, hist_tree, hist_loss


os.environ['CUDA_VISIBLE_DEVICES'] = str()

torch.manual_seed(123)

L = 6

X1 = torch.randint(0,4,(L,2)).double()
X2 = torch.randint(0,4,(L,2)).double()

X1_orig = X1.clone().detach()

X1.requires_grad = True

n_bins = 4
n_layers = 3
sigma = 0.01
interp = 'raised cos'
n_steps = 200
lamb = 0
optim_params = {'lr': 0.001, 'momentum': 0.9}

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
for X_i in X_arr: plt.plot(*X_i)

X1_orig_np = X1_orig.detach().numpy().T
X1_np = X1.detach().numpy().T
X2_np = X2.detach().numpy().T

plt.plot(*X1_orig_np,'k+',label='Source')
plt.plot(*X1_np,'kx',label='Output')
plt.plot(*X2_np,'ko',fillstyle='none',label='Target')

plt.legend()
plt.tight_layout()

plt.show()



