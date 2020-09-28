import numpy as np
from PIL import Image
import io

class RandomCrop:
	def __init__(self,L): self.L = L
	def __str__(self): return f'Random Crop {self.L}x{self.L}'
	def __call__(self,X):
		h,w,_ = X.shape
		ih = np.random.randint(h-self.L) if h>L else 0
		iw = np.random.randint(w-self.L) if w>self.L else 0
		return X[ih:,iw:][:self.L,:self.L]

class CenterCrop:
	def __init__(self,L): self.L = L
	def __str__(self): return f'Center Crop {self.L}x{self.L}'
	def __call__(self,X):
		h,w,_ = X.shape
		ih = max(0,(h-self.L)//2)
		iw = max(0,(w-self.L)//2)
		return X[ih:,iw:][:self.L,:self.L]

class JPEGFilter:
	def __init__(self,Q): self.Q = Q
	def __str__(self): return f'JPEG Filter Q = {self.Q}'
	def __call__(self,X):
		if self.Q is None: return X
		else:
			im_in = Image.fromarray(X)
			im_buffer = io.BytesIO()
			im_in.save(im_buffer,'JPEG',quality=self.Q)
			im_out = Image.open(im_buffer)
			return np.array(im_out)

class Identity:
	def __str__(self): return 'Identity'
	def __call__(self,X): return X


def hwc2chw(X):
	if len(X.shape) == 4: return X.transpose((0,3,1,2))
	else: return X.transpose((2,0,1))

def chw2hwc(X):
	if len(X.shape) == 4: return X.transpose((0,2,3,1))
	else: return X.transpose((1,2,0))



if __name__=='__main__':

	import matplotlib.pyplot as plt
	import matplotlib

	np.random.seed(123)
	img = Image.open(matplotlib.cbook.get_sample_data('grace_hopper.png'))
	X = np.array(img)
	
	f_list = [Identity(), RandomCrop(256), CenterCrop(256), CenterCrop(384),
                  JPEGFilter(None), JPEGFilter(95), JPEGFilter(75), JPEGFilter(50)]

	L = len(f_list)
	w = int(np.ceil(np.sqrt(L)))
	h = int(np.ceil(L/w))
	fig, axes = plt.subplots(h,w)

	for a, f in zip(axes.reshape(-1),f_list):
		a.imshow(f(X))
		a.set_title(f)

	plt.tight_layout()
	plt.show()

