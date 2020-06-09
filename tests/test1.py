import sys, os
sys.path.append('../utils/')
from co_occur import RaisedCos, Hist
import torch
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = str()

X1 = torch.tensor([0,0,1,2]).view(-1,1).double()
X2 = torch.tensor([0,1,2,2]).view(-1,1).double()


X1.requires_grad = True


n_bins = 4
sigma = 0.001
interp = RaisedCos()
n_steps = 1000
optim_params = {'lr': 0.001, 'momentum': 0.9}


H2_1 = Hist.apply(X2,0,n_bins,interp)
H2_2 = Hist.apply(X2/2,0,n_bins//2+1,interp)
#print(H2_2)
#print(X2)
#quit()


#print(X2)
#print(X2/2)
#quit()

optimizer = torch.optim.SGD((X1,),**optim_params)

X_list = list()

for sigma in [0.0001,0]:

	for i in range(n_steps):

		optimizer.zero_grad()
		X1.data = X1.data + sigma*torch.randn(X1.shape).double()
		X1_clamp = torch.clamp(X1,0,n_bins-1)

		X_list.append(X1_clamp.detach().numpy())

		H1_1 = Hist.apply(X1_clamp,0,n_bins,interp)
		H1_2 = Hist.apply(X1_clamp/2,0,n_bins//2+1,interp)	

		loss = torch.sum(torch.abs(H1_1-H2_1)) + 2*torch.sum(torch.abs(H1_2-H2_2))

	#	H1 = Hist.apply(X1_clamp,sigma,n_bins,interp)
	#	loss = torch.sum(torch.abs(H1-H2))
		loss.backward()
		optimizer.step()

		#print('\n\nStep',i,loss,X1)
		#print(H1_1,H2_1)
		#print(H1_2,H2_2)

#print(X_list)
X_arr = np.array(X_list).squeeze(2).T

print(X_arr.shape)

for X_i in X_arr: plt.plot(X_i)
#plt.plot(X_arr[0])

plt.show()
print('hello world')


