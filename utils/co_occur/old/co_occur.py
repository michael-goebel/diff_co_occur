
from my_alg import img2pairs
from hist import Hist


def co_occur(X): return Hist.apply(img2pairs(X),n_bins=256,'raised cos')

def co_occur_w_norm(X):
	X_out = co_occur(X).astype('float')
	X_out /= X_out.max((2,3))
	









