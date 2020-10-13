import torch

from utils.co_occur.hist import Hist
from utils.co_occur.gbco import img2pairs, img2pairs_cband


eps_log = 10**(-6)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


class CoOccurWithNorm(torch.nn.Module):
	def __init__(self,cband=False):
		super(CoOccurWithNorm,self).__init__()
		self.pairs_func = img2pairs_cband if cband else img2pairs
	def forward(self,X):
		C = torch.stack([torch.stack([Hist.apply(pairs,256,'raised cos') for pairs in self.pairs_func(X_i,ch_first=True)]) for X_i in X])
		v_max = torch.max(torch.max(C,2)[0],2)[0].float()
		C_out = C.float() / v_max[:,:,None,None]
		return C_out


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

