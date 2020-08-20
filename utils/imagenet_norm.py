import numpy as np


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def imagenet_norm(X):
	
	assert len(X.shape)==3, print('input should be a 3 dimensional array')
	if X.shape[0] == 3: return ((X.astype('float')/255) - mean[:,None,None])/std[:,None,None]
	elif X.shape[2] == 3: return ((X.astype('float')/255) - mean[None,None,:])/std[None,None,:]
	else: print('Error, not hwc or chw'); quit()



