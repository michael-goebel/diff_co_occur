import numpy as np
from torch.utils.data import DataLoader

from utils.image.image_io import image_reader
from utils.image.pre_proc import CenterCrop
from utils.detection.file_fetcher import get_files


class SingleDataGen:
	def __init__(self,files,labels,groups,pre_proc):
		self.files = files; self.labels = labels; self.groups = groups
		self.pre_proc = [image_reader,] + pre_proc
	
	def __len__(self): return len(self.files)

	def __getitem__(self,i):
		print(i)
		x = self.files[i]
		for f in self.pre_proc: x = f(x)
		return x.astype('float'), self.labels[i], self.groups[i]


def stack_data_lists(f_lists, label_lists):
	counts = np.array([len(f) for f in f_lists])
	labels = np.hstack([l*np.ones(len(f)) for l,f in zip(label_lists,f_lists)])
	groups = np.hstack([i*np.ones(len(f)) for i,f in enumerate(f_lists)])
	files = sum(f_lists,[])
	inds = np.split(np.arange(len(files)),np.cumsum(counts)[:-1])
	print(labels.shape)
	return files, labels, groups, inds


class BatchSampler:
	def __init__(self,inds_list,bs_list):
		self.inds = inds_list
		self.bs_arr = np.array(bs_list)
		self.L = np.min(np.array([len(i) for i in self.inds]) // self.bs_arr)
		self.shuffle()

	def shuffle(self):
		self.inds_out = np.hstack([np.random.permutation(i)[:n*self.L].reshape((self.L,n)) for i,n in zip(self.inds,self.bs_arr)])
		print(self.inds_out[0].shape)

	def __len__(self): return self.L
	def __iter__(self): return iter(self.inds_out)


def get_data_gen(data_group,tvt,pre_proc,n_cpu=1):

	f_lists = get_files(data_group,tvt)
	n_fake = len(f_lists) - 1
	l_labels = [0,] + [1,]*n_fake

	bs_real = 20 if data_group == 'all_adv' else 16
	bs_list = [bs_real,] + [bs_real//n_fake]*n_fake
	

	files, labels, groups, inds = stack_data_lists(f_lists,l_labels)

	flat_data_gen = SingleDataGen(files,labels,groups,pre_proc)
	
	
	if tvt == 'train':
		return DataLoader(flat_data_gen, batch_sampler=BatchSampler(inds,bs_list))

	else:
		return DataLoader(flat_data_gen, batch_size = sum(bs_list))


