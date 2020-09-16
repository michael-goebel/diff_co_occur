
import sys
sys.path.append('../co_occur/utils')


#from my_alg import img2pairs
from hist import Hist
import torch


eps_log = 10**(-6)
print(eps_log)

# assumes channels first
def img2pairs(X): return [torch.stack((Xi[:-1].reshape(-1),Xi[1:].reshape(-1)),dim=1) for Xi in X]



rgb_pairs = [[0,1],[0,2],[1,2]]

def img2pairs_cband(X):
	out = [torch.stack((X[i,:-1,:-1].reshape(-1),X[i,1:,1:].reshape(-1)),dim=1) for i in range(3)]
	out += [torch.stack((X[i].reshape(-1),X[j].reshape(-1)),dim=1) for i,j in rgb_pairs]
#        out = [torch.stack((X[:-1,:-1,i].reshape(-1),X[1:,1:,i].reshape(-1)),dim=1) for i in range(3)]
#        out += [torch.stack((X[:,:,i].reshape(-1),X[:,:,j].reshape(-1)),dim=1) for i,j in rgb_pairs]
	#print([i.shape for i in out])
	return out



def co_occur(X): return Hist.apply(img2pairs(X),256,'raised cos')

def co_occur_w_norm(X):
	X_out = co_occur(X).astype('float')
	X_out /= X_out.max((2,3))

def dft_w_norm(X):
	dft = torch.rfft(X,signal_ndim=2,normalized=True)
	dft_mag = torch.norm(dft,dim=-1)
	return dft_mag

def fftshift2d(X):
	roll_shift = tuple(i//2 for i in X.shape[-2:])
	print(roll_shift)
	return torch.roll(X, roll_shift, (-2,-1))


imagenet_mean = [0.485, 0.456, 0.406] 
imagenet_std = [0.229, 0.224, 0.225]

def imagenet_norm(X):
	mean = torch.Tensor(imagenet_mean).to(X.device)
	std = torch.Tensor(imagenet_std).to(X.device)
	return	((X/255) - mean[None,:,None,None]) / std[None,:,None,None]

	

class CoOccurWithNorm(torch.nn.Module):
	def forward(self, X):
		C = torch.stack([torch.stack([Hist.apply(pairs,256,'raised cos') for pairs in img2pairs(X_i)]) for X_i in X])
		v_max = torch.max(torch.max(C,2)[0],2)[0]
		C_out = C.float() / v_max[:,:,None,None]
		return C_out

class CBandCC(torch.nn.Module):
	def forward(self, X):
		C = torch.stack([torch.stack([Hist.apply(pairs,256,'raised cos') for pairs in img2pairs_cband(X_i)]) for X_i in X])
		v_max = torch.max(torch.max(C,2)[0],2)[0]
		C_out = C.float() / v_max[:,:,None,None]
		return C_out



#Hist.apply(img2pairs(X),256,'raised cos').astype('float')
#		C_out /= C.max((2,3))
#		return C_out


class DFTWithNorm(torch.nn.Module):
	def forward(self,X):
		dft = torch.rfft(X,signal_ndim=2,normalized=True,onesided=False)
		dft_mag = torch.norm(dft,dim=-1)
		dft_log = torch.log(eps_log + dft_mag)
		dft_positive = dft_log - dft_log.min()
		dft_normed = dft_positive / dft_positive.max()
		return 2*dft_normed - 1


class ImageNetNorm(torch.nn.Module):
	def forward(self,X):
		mean = torch.Tensor(imagenet_mean).to(X.device)
		std = torch.Tensor(imagenet_std).to(X.device)
		return	((X/255) - mean[None,:,None,None]) / std[None,:,None,None]






