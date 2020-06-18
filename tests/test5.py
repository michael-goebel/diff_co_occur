from PIL import Image
import numpy as np
import torch
import sys, os
sys.path.append('../utils/')
from hist import RaisedCos, L1Dist, Hist, hist_tree, hist_loss
import matplotlib.pyplot as plt


def img2pairs(X):
	h,w = X.shape
	return torch.stack((X[:-1,:].view(-1),X[1:,:].view(-1)),dim=1)


def img_stats(X): return np.mean(X,(0,1)), np.std(X,(0,1))



os.environ['CUDA_VISIBLE_DEVICES']='2'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}')

real_img = '/home/mike/spring_2020/gan_images/real/n02381460_1108_real_A.png'
fake_img = '/home/mike/spring_2020/gan_images/fake/n02381460_1019_fake_B.png'

n_bins = 256
n_layers = 8
n_steps = 300
n_print = 10
sigma = 0.01
lamb = 0
interp = RaisedCos()
optim_params = {'lr': 0.01, 'momentum': 0.9}
gs_kw = {'cmap':'gray', 'vmin':0, 'vmax':255}


L = 256

I1_np = np.array(Image.open(fake_img),dtype='double')[:L,:L,1]
I2_np = np.array(Image.open(real_img),dtype='double')[:L,:L,1]

#I1_np = 2 * np.ones((L,L),dtype='double')
#I2_np = 1 * np.ones((L,L),dtype='double')



#fig, axes = plt.subplots(2,2)

#axes[0,0].imshow(I1_np,**gs_kw)
#axes[1,0].imshow(I2_np,**gs_kw)



I1 = torch.tensor(I1_np).double().to(device)
I2 = torch.tensor(I2_np).double().to(device)

I1_orig = I1.clone().detach()

I1.requires_grad = True


X2 = img2pairs(I2)
#print(X2)

H2 = hist_tree(X2,Hist,n_bins,n_layers,interp)

#fig, axes = plt.subplots(2,4)
#axes = axes.reshape(-1)

optimizer = torch.optim.SGD((I1,),**optim_params)

for i in range(n_steps):

	optimizer.zero_grad()
	noise = sigma*torch.randn(I1.shape).double().to(device)
	I1.data = torch.clamp(I1+noise,0,n_bins-1)
	X1 = img2pairs(I1)
	H1 = hist_tree(X1,Hist,n_bins,n_layers,interp)
	loss = hist_loss(H1,H2)
	loss.backward()
	mse = torch.mean((I1-I1_orig)**2)
	if i % n_print == 0: print(f'Loss: {loss}\nMSE: {mse}\n')

	optimizer.step()



labels_list = ['Source', 'Solution', 'Target']

X_list = [I.detach().cpu() for I in [I1_orig, I1, I2]]

C_list = [Hist.apply(img2pairs(X),n_bins,interp) for X in X_list]


fig,axes = plt.subplots(2,3)

for a, X in zip(axes[0],X_list): a.imshow(X.numpy(),**gs_kw)
for a, C in zip(axes[1],C_list): a.imshow(np.log(1+C.numpy()))

for a, l in zip(axes[0],labels_list): a.set_title(l)


plt.show()





