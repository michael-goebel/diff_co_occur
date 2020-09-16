import numpy as np
import sys
sys.path.append('../../utils/')
from image_reader import image_reader
from pre_proc import CenterCrop



class TrainDataGen:

	def __init__(self,files_lists,file_labels,bs_list,n_batch_max,pre_proc_funcs):
		self.files_lists = files_lists
		self.n_files = np.array([len(f_list) for f_list in files_lists])
		self.file_labels = np.array(file_labels)
		self.npb = np.array(bs_list)
		self.n_class = max(file_labels) + 1
		self.pre_proc_funcs = [image_reader,] + pre_proc_funcs

		self.L = np.min(self.n_files//self.npb)
		if not n_batch_max is None: self.L = min(n_batch_max,self.L)

	def __len__(self): return self.L		

	def __iter__(self):
		self.i = 0
		self.inds = [np.random.permutation(n) for n in self.n_files]
		return self


	def __next__(self):
		if self.i < self.L:
			inds_list = [ind[b*self.i:b*(self.i+1)] for ind,b in zip(self.inds,self.npb)]
			x_list = [f[ind_i] for f, ind_l in zip(self.files_lists,inds_list) for ind_i in ind_l]
			for func in self.pre_proc_funcs: x_list = [func(x_i) for x_i in x_list]
			y = np.hstack([[l,]*b for l,b in zip(self.file_labels,self.npb)])
			self.i += 1
			return np.array(x_list), np.array(y)

		else:
			raise StopIteration()

class TestDataGen:
	
	def __init__(self,files_lists,file_labels,batch_size,pre_proc_funcs):
		self.files = sum(files_lists,[])
		self.parts = np.hstack([i*np.ones(len(f)) for i,f in enumerate(files_lists)])
		self.labels = np.array(file_labels)[self.parts]
		self.bs = batch_size
		self.pre_proc_funcs = [image_reader,] + pre_proc_funcs

	def __len__(self): return int(np.ceil(len(self.files)/self.bs))

	def __iter__(self):
		self.i = 0
		return self

	def __next__(self):
		if self.i < self.L:
			y = self.labels[self.bs*i:self.bs*(i+1)]
			x_list = x_list[self.bs*i:self.bs*(i+1)]
			for func in self.pre_proc_funcs: x_list = [func(x_i) for x_i in x_list]
			self.i += 1	
			return np.array(x_list), y




if __name__=='__main__':


	data_dir = '/media/ssd1/mike/gan_data_trimmed/split_files/'
	f1_txt = data_dir + 'reg_train_real.txt'
	f2_txt = data_dir + 'reg_train_fake.txt'

	def read_lines(fname):
		with open(fname) as f: return f.read().split('\n')

	f_list = [read_lines(f) for f in [f1_txt,f2_txt]]

	dg = TrainDataGen(f_list,[0,1],[4,4],4,[CenterCrop(256)])
	print(len(dg))

	for X,y in dg:
		print(X.shape,y)




#def __init__(self,files_lists,file_labels,bs_list,n_batch_max,pre_proc_funcs):





