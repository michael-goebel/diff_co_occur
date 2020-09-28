import numpy as np
from utils.image.image_io import image_reader
from utils.image.pre_proc import CenterCrop


class SingleDataGen:
	def __init__(self,files,labels,pre_proc):
		self.files = files; self.labels = labels
		self.pre_proc = [image_reader,] + pre_proc
	
	def __len__(self): return len(self.files)

	def __getitem__(self,i):
		x = self.files[i]
		for f in self.pre_proc: x = f(x)
		return x, self.labels[i]


def stack_data_lists(f_lists, label_lists):
	counts = np.array([len(f) for f in f_lists])
	labels = np.hstack([l*len(f) for l,f in zip(label_lists,f_lists)])
	files = sum(files,[])
	inds = np.split(np.arange(len(files)),np.cumsum(counts)[:-1])
	return files, labels, inds


class BatchSampler:
	def __init__(self,inds_list,bs_list):
		self.inds = self.inds_list
		self.bs_arr = np.array(bs_list)
		self.L = np.min(np.array([len(i) for i in self.inds]) // self.bs_arr)
		self.shuffle()

	def shuffle(self):
		self.inds_out = [np.random.permutation(i)[:n*L].reshape((n,L)) for i,n in zip(self.inds,bs_arr)]

	def __len__(self): return self.L
	def __iter__(self): return iter(self.inds_out)


