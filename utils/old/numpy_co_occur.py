import numpy as np

def ub(x): return min(x,0) or None
def lb(x): return max(x,0)

pair2list = {'h':[(0,1)], 'v':[(1,0)], 'hv':[(0,1),(1,0)], 'hvd':[(0,1),(1,1),(1,0),(1,-1)]}



def co_occur_unit(X,dy,dx,L):
    X1 = X[lb(dy):ub(dy),lb(dx):ub(dx)]         # array of first indices for co-occur pair
    X2 = X[lb(-dy):ub(-dy),lb(-dx):ub(-dx)]	# array of second args
    pairs = (X1 + L*X2).reshape(-1)		# pack the tuple of two into one int
    return np.bincount(pairs,minlength=L**2).reshape((L,L))


class CoOccur:
	def __init__(self,pairs,chan_first,L): self.pairs=pairs; self.c_first=chan_first; self.L=L
	def __str__(self): return f'Co Occur ({self.pairs})'
	def __call__(self,X):
		X_chw = X if self.c_first else np.rollaxis(X,2)
		co_list = [co_occur_unit(X_i,*d,self.L) for d in pair2list[self.pairs] for X_i in X_chw]
		return np.stack(co_list,axis=0 if self.c_first else 0)

def co_occur_normalize(X): return X.astype('float') / np.max(X,(1,2))[:,None,None]


if __name__=='__main__':

	np.random.seed(123)
	h, w, c = 4, 5, 3
	n_max = 4
	X = np.random.randint(0,n_max,(c,h,w))

	print('Input')
	print(X)

	cc = CoOccur(pairs='h',chan_first=True,L=n_max)
	C = cc(X)

	print('Co Occurrence')
	print(C)
	print(C.shape)
	print(co_occur_normalize(C))
