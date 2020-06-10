import sys, os
sys.path.append('../utils/')
#from co_occur import RaisedCos, Hist, L1Dist
from hist import RaisedCos, L1Dist, Hist, hist_tree, hist_loss
#from hist_other import hist_tree, hist_loss



import torch
import numpy as np
import matplotlib.pyplot as plt



#def hist_tree(X,H,n_bins,n_layers,interp):
#	return [H.apply(X/(2**i),n_bins//(2**i)+1,interp) for i in range(n_layers)]

#def hist_loss(H1,H2): return sum([(2**i)*torch.sum(torch.abs(h1-h2)) for i,(h1,h2) in enumerate(zip(H1,H2))])




os.environ['CUDA_VISIBLE_DEVICES'] = str()

X1 = torch.tensor([0,0,1,2,3,4,5,6]).view(-1,1).double()
X2 = torch.tensor([0,1,2,3,4,5,6,6]).view(-1,1).double()


X1.requires_grad = True


n_bins = 8
n_layers = 4
sigma = 0.001
interp = RaisedCos()
#interp = L1Dist()
n_steps = 200
optim_params = {'lr': 0.001, 'momentum': 0.9}


H2 = hist_tree(X2,Hist,n_bins,n_layers,interp)


#H2_1 = Hist.apply(X2,n_bins,interp)
#H2_2 = Hist.apply(X2/2,n_bins//2+1,interp)
#print(H2_2)
#print(X2)

#print(hist_tree(X2,Hist,n_bins,2,interp))

#quit()


#print(X2)
#print(X2/2)
#quit()

optimizer = torch.optim.SGD((X1,),**optim_params)

X_list = list()


for i in range(n_steps):

	optimizer.zero_grad()
	X1.data = X1.data + sigma*torch.randn(X1.shape).double()
	X1_clamp = torch.clamp(X1,0,n_bins-1)

	X_list.append(X1_clamp.detach().numpy())

#	H1_1 = Hist.apply(X1_clamp,n_bins,interp)
#	H1_2 = Hist.apply(X1_clamp/2,0,n_bins//2+1,interp)	
	H1 = hist_tree(X1_clamp,Hist,n_bins,n_layers,interp)

	#loss = torch.sum(torch.abs(H1_1-H2_1)) + 2*torch.sum(torch.abs(H1_2-H2_2))

#	loss = sum([(2**i)*torch.sum(torch.abs(h1-h2)) for i,(h1,h2) in enumerate(zip(H1,H2))])

	loss = hist_loss(H1,H2)

	loss.backward()
	optimizer.step()

#print(X_list)
X_arr = np.array(X_list).squeeze(2).T

print(X_arr.shape)

for X_i in X_arr: plt.plot(X_i)
#plt.plot(X_arr[0])

plt.show()
print('hello world')


