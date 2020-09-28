import torch
import numpy as np

n = 6
m = 2

X = np.random.randint(0,100,n)

class Batcher:
	def __init__(self): self.shuffle()
	def shuffle(self):
		self.inds = np.random.permutation(n).reshape((-1,m))
	def __iter__(self): return iter(self.inds)
	def __len__(self): return self.inds.shape[0]


#batcher = Batcher()

dg = torch.utils.data.DataLoader(X,batch_sampler=Batcher())
print(len(dg))		


for i in range(3):

	print(list(dg))
	#batcher.shuffle()
	dg.batch_sampler.shuffle()
print(dir(dg.batch_sampler))
print(dir(dg))


