import numpy as np
from scipy.spatial import distance

from utils.image.image_io import image_reader
from utils.image.pre_proc import CenterCrop


L = 256


def img2hist(X): return np.array([np.bincount(X[:,:,i].reshape(-1),minlength=L) for i in range(3)]).T

def apply_funcs(x,funcs):
	y = x
	for f in funcs: y = f(y)
	return y


def get_dists(f_list1, f_list2, pre_proc_list=list()):

	h_arr1 = np.empty((len(f_list1),L,3),dtype='int')
	h_arr2 = np.empty((len(f_list2),L,3),dtype='int')

	all_funcs = [image_reader, *pre_proc_list, img2hist]

	for i, f in enumerate(f_list1): h_arr1[i] = apply_funcs(f,all_funcs)
	for i, f in enumerate(f_list2): h_arr2[i] = apply_funcs(f,all_funcs)

	cdf_arr1 = np.cumsum(h_arr1,axis=1)
	cdf_arr2 = np.cumsum(h_arr2,axis=1)

	D = distance.cdist(cdf_arr1.reshape((-1,3*L)), cdf_arr2.reshape((-1,3*L)), metric='cityblock')
	D /= (3*(L**2))

	return D


def get_pairs(f_src, f_tgt, pre_proc_list=list()):

	D = get_dists(f_src, f_tgt, pre_proc_list)
	f_tgt_inds = np.argmin(D,axis=1)
	f_tgt_match = [f_tgt[i] for i in f_tgt_inds]
	return list(zip(f_src, f_tgt_match))



if __name__=='__main__':

	import matplotlib.pyplot as plt

	real_file = '/home/mike/spring_2020/prelim_cc_tests/train_val_splits/val_real.txt'
	fake_file = '/home/mike/spring_2020/prelim_cc_tests/train_val_splits/val_fake.txt'
	pre_proc_funcs = [CenterCrop(L),]

	with open(real_file) as f: real_list = f.read().split('\n')
	with open(fake_file) as f: fake_list = f.read().split('\n')

	print(get_pairs(fake_list[:30], real_list[:15], pre_proc_funcs))



