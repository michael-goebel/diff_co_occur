import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../utils/')
from hist import RaisedCos, L1Dist, Hist, hist_tree, hist_loss
from itertools import permutations
from tqdm import trange

os.environ['CUDA_VISIBLE_DEVICES'] = str()

torch.manual_seed(123)

L = 7

n_trials = 100

#X1_orig = torch.randint(0,4,(L,2)).double()
#X2_orig = torch.randint(0,4,(L,2)).double()


n_bins = 8
interp = 'raised cos'
n_steps = 500
optim_params = {'lr': 0.001, 'momentum': 0.9}


def emd(X1,X2):
	P1 = np.array(list(permutations(X1)))
	P1 -= X2[None,:,:]
	P1 = np.abs(P1)
	dists = np.sum(P1,(1,2))
	#print(X1,X2)

	#return 1.2
	return dists.min()




def run(X1_orig,X2_orig,n_steps,sigma,n_layers):

	X1 = X1_orig.clone()
	X2 = X2_orig.clone()
	X1.requires_grad = True
	H2 = hist_tree(X2,n_bins,n_layers,interp)	
	optimizer = torch.optim.SGD((X1,),**optim_params)

	for i in range(n_steps):
		optimizer.zero_grad()
		X1.data = torch.clamp(X1+sigma*torch.randn(X1.shape).double(),0,n_bins-1)
		H1 = hist_tree(X1,n_bins,n_layers,interp)
		loss = hist_loss(H1,H2)
		loss.backward()
		optimizer.step()

	return np.round(X1.detach().numpy()).astype('int')

#print(X1_orig)
#print(X2_orig)

#1 = np.array(X1_orig).astype('int')
#2 = np.array(X2_orig).astype('int')


for param_set in [[0.01,1],[0.0,3],[0.01,3]]:

	suc_list = list()
	opt_list = list()
	for i in trange(n_trials):
		X1_orig = torch.randint(0,4,(L,2)).double()
		X2_orig = torch.randint(0,4,(L,2)).double()

		X1 = np.array(X1_orig).astype('int')
		X2 = np.array(X2_orig).astype('int')



		X3 = run(X1_orig,X2_orig,n_steps,*param_set)
		suc_list.append(emd(X2,X3)==0)
		opt_list.append((emd(X1,X2) == emd(X1,X3)) & (emd(X2,X3)==0) )
	print(sum(suc_list), sum(opt_list))




#X4 = run(X1_orig,X2_orig,n_steps,0.00,3)
#X5 = run(X1_orig,X2_orig,n_steps,0.001,3)

#X2 = np.array(X2_orig).astype('int')



#print(emd(X3,X2))
#print(emd(X4,X2))
#print(emd(X5,X2))



#print(run(X1_orig,X2_orig,n_steps,0.001,1))
#print(run(X1_orig,X2_orig,n_steps,0.00,3))
#print(run(X1_orig,X2_orig,n_steps,0.001,3))

#emd(X1_orig.numpy(),None)

#
