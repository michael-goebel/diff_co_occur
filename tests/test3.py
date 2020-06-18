from PIL import Image
import numpy as np
import torch
import sys
sys.path.append('../utils/')
from hist import RaisedCos, L1Dist, Hist, hist_tree, hist_loss


def img2pairs(X):
	h,w,c = X.shape
	return [torch.stack((X[:-1,:,i].view(-1),X[1:,:,i].view(-1)),dim=1) for i in range(c)]

def img_stats(X): return np.mean(X,(0,1)), np.std(X,(0,1))



real_img = '/home/mike/spring_2020/gan_images/real/n02381460_1108_real_A.png'
fake_img = '/home/mike/spring_2020/gan_images/fake/n02381460_1019_fake_B.png'

n_bins = 256
n_layers = 8
n_steps = 100
sigma = 0.01
lamb = 0
interp = RaisedCos()
optim_params = {'lr': 100.0, 'momentum': 0.9}


I1_np = np.array(Image.open(fake_img),dtype='double')
I2_np = np.array(Image.open(real_img),dtype='double')



#print(img_stats(I1_np))
#print(I1_np.dtype)

I1_stats = img_stats(I1_np)
I2_stats = img_stats(I2_np)

I1_np = (I1_np - I1_stats[0])/I1_stats[1]
I1_np = I1_np * I2_stats[1] + I2_stats[0]


print(I1_np.shape)
print(I2_np.shape)

I1 = torch.tensor(I1_np).double()
I2 = torch.tensor(I2_np).double()

I1_orig = I1.clone().detach()

I1.requires_grad = True


X2 = img2pairs(I2)

print(len(X2))
print(X2[0].shape)

#print(X2)
#quit()

H2 = [hist_tree(X,Hist,n_bins,n_layers,interp) for X in X2]
#print(H2)
#print(len(H2))
#print(len(H2[0]))
optimizer = torch.optim.SGD((I1,),**optim_params)

for i in range(n_steps):

	optimizer.zero_grad()
	I1.data = torch.clamp(I1+sigma*torch.randn(I1.shape).double(),0,n_bins-1)

	X1 = img2pairs(I1)
	H1 = [hist_tree(X,Hist,n_bins,n_layers,interp) for X in X1]
	loss = sum([hist_loss(h1,h2) for h1,h2 in zip(H1,H2)]) + lamb*torch.sum((I1-I1_orig)**2)
	print(loss)
	print(torch.sum((I1-I1_orig)**2))
	optimizer.step()

#X1 = img2pairs(I1)
#X2 = img2pairs(


#print(img2pairs(I1))

#print(I1.shape)






