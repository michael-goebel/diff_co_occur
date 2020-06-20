import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import os

pi = torch.tensor(math.pi)


def hist_tree(X,H,n_bins,n_layers,interp):
        return [H.apply(X/(2**i),n_bins//(2**i)+1,interp) for i in range(n_layers)]

def hist_loss(H1,H2): return sum([(2**i)*torch.sum(torch.abs(h1-h2)) for i,(h1,h2) in enumerate(zip(H1,H2))])


# Special 1D function class, also needs f(x) and df(x) (the derivative of f) to be defined
# Should be an interpolation function, which is non-zero only where abs(x) < bound or abs(x) <= bound
# "bound_inc" dictates whether or not bound is inclusive or exclusive
class MyFunc:
	bound = 1
	bound_inc = True
	neigh_dtype = torch.float32	# non-ideal situation, labels must be packed using matmul, which only accepts float on cuda

	def mask(self,x):	# return a mask, indicating for each point if it is in the nonzero neighborhood
		if self.bound_inc: return torch.abs(x) <= self.bound
		else: return torch.abs(x) < self.bound

	def neighs(self,x):	# return list of neighbors for each point
		b = math.ceil(self.bound)
		return [torch.floor(x-i).type(self.neigh_dtype) for i in range(-b,b+1)]

class RaisedCos(MyFunc):
	def f(self,x): return (1+torch.cos(x*pi))/2*(self.mask(x).type(x.dtype))
	def df(self,x): return (-pi/2)*torch.sin(x*pi)*(self.mask(x).type(x.dtype))

class L1Dist(MyFunc):
	def f(self,x): return (1-torch.abs(x))*( (x < self.bound).type(x.dtype) + (x <= self.bound).type(x.dtype))/2
	def df(self,x): return torch.sign(-x)*((torch.abs(x) < self.bound).type(x.dtype)+(torch.abs(x) <= self.bound).type(x.dtype))/2


default_interp = RaisedCos()


class Hist(torch.autograd.Function):

	"""
	ND differentiable histogram function
	x: Input tensor of size (M,N). Each of the M rows is an input tuple representing an N-dimensional index
	sigma: standard deviation for added noise
	n_bins: number of bins (must be the same for each dimension). Output will be of shape (n_bins,)*N
	func: Like MyFunc, a class defined in this same file. Used to determine 1D interpolation function
	K: number of neighbors in ND space
	M: input length
	N: number of dimensions
	P: number of neighbors in 1D space
	"""

	@staticmethod
	def forward(ctx,x,n_bins,func=default_interp):
		M, N = x.shape
		clamped_input = torch.clamp(x,0,n_bins-1)
		labels = torch.stack(func.neighs(clamped_input))	# This stack gives P 1D neighbors for each dimension
		P = labels.shape[0]
		labels = labels.transpose(2,1)

		inds0 = torch.arange(P**N)[:,None] // (P**torch.arange(N))[None,:] % P	# inds0 is indexing array, only used in next line
		labels = labels[inds0,torch.arange(N)[None,:]]	# Get all P**N possible combinations neighbors

		labels = labels.transpose(0,2).transpose(1,2)	# premutes axes
		K = labels.shape[1]

		inds_invalid = ((labels < 0) | (labels >= n_bins)).any(2)	# by default consider all neighbors. Then check if they are outside valid range
		labels[inds_invalid] = 0					# if invalid, replace with some valid index (I close 1). Later on, these invalid
										# indices must be remembered, and values associated with them should not be counted

		# Below: pack tuple of ints into single int. Ex, if n_bins is 4, N=3, then (1,2,3) -> 1*16 + 2*4 + 3*1 = 27, like np.ravel_multi_index
		packing_strides = (n_bins**torch.arange(N-1,-1,-1)).type(labels.dtype).to(labels.device)
		labels_pack = torch.matmul(labels.reshape(M*K,N), packing_strides).view(M,K).long()

		deltas = x[:,None,:] - labels.type(x.dtype)	# For each neighbor, compute distance along each dimension
		f_d = func.f(deltas)				# Apply forward interpolation function
		df_d = func.df(deltas)				# Apply derivative interpolation function (not used in forward but saved for backward)

		f_d[inds_invalid] = 0		# since computations of output are all summations, zeroing the values associated with invalid
		df_d[inds_invalid] = 0		# indices will "remove" them, while maintaining fixed tensor shape

		scores = f_d.prod(2)		# ND hist interpolation values are defined as product of 1D values, like bilinear interpolation

		# Next line is like scipy.ndimage.sum
		# After this, the histogram is 1D and the bins represent the packed labels. The view call unpacks/reshapes into the N-dim histogram we expect.
		hist = torch.bincount(labels_pack.view(-1),weights=scores.view(-1),minlength=n_bins**N).view((n_bins,)*N)
		ctx.save_for_backward(f_d,df_d,labels_pack)
		return hist

		
	@staticmethod
	def backward(ctx, backprop_tens):	# computes jacobian-vector product
		f_d, df_d, labels_pack = ctx.saved_tensors	# load saved tensors from forward
		M,K,N = f_d.shape
		grads = torch.stack((f_d,)*N)		# These four lines compute gradients
		inds = torch.arange(N)			# Uses product rule
		grads[inds,:,:,inds] = df_d.transpose(0,2).transpose(1,2)
		grads = grads.prod(3)

		# grads and labels together represent a sparse encoding of the jacobian matrix
		# The list lines compute jacobian-vector product
		grads *= backprop_tens.view(-1)[labels_pack][None,:,:]	# 
		grads = grads.sum(2)
		grads = grads.transpose(0,1)
		return (grads,None,None,None)



# Brute force nd histogram for debugging and checking efficient method. Does not utilize the fact that only a small number of nearby points are nonzero.
def hist_brute_force(x,n_bins,func):
	M,N = x.shape
	deltas = torch.stack( (x.double(),)*(n_bins**N) ).view( ([n_bins,] * N) + [M,N])
	inds = (torch.arange(n_bins**N)[:,None] // (n_bins**torch.arange(N-1,-1,-1))[None,:] % n_bins).view(([n_bins,]*N) + [1,N])
	deltas -= inds.double()
	f_d = func.f(deltas)
	return f_d.prod(-1).sum(-1)

# Simplest function for checking integer intputs. Packs labels, uses bincount, then unpacks.
def hist_integer(x,n_bins):
	M,N = x.shape
	l = torch.matmul(x.long(), (n_bins**torch.arange(N-1,-1,-1)).long())
	return torch.bincount(l,minlength=(n_bins**N)).view((n_bins,)*N)



if __name__ == '__main__':


	interp = RaisedCos()
	#interp = L1Dist()
	torch.manual_seed(123)

	n_bins = 4
	M = 12
	N = 2

	X1 = torch.randint(0,n_bins,(M,N)).double()
	X2 = (n_bins-1)*torch.rand(M,N).double()
	X3 = X2[:,:1]

	X_list = [X1,X2,X3]

	hist_list = [
		lambda x: hist_integer(x,n_bins),
		lambda x: hist_brute_force(x,n_bins,interp),
		lambda x: Hist.apply(x,n_bins,interp)
	]
	names = ['int_cast', 'brute_force', 'efficient']

	for X in X_list: X.requires_grad = True

	for X in X_list:

		print('\n\nInput:\n',X)
		for n,h in zip(names,hist_list):
			print(n)
			print(h(X))
	torch.set_printoptions(precision=10)
	for i,X in enumerate(X_list):

		print(f'\nGradient check input {i+1}')
		for n,h in zip(names[1:],hist_list[1:]):
			print(f'{n}: {torch.autograd.gradcheck(h,(X,),raise_exception=False)}')


