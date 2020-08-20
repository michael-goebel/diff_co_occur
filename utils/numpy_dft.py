import numpy as np
from image_reader import image_reader


def normed_dft(X,eps=0.01):
	assert len(X.shape)==3, print('Input should be hwc or chw')
	if X.shape[-3] == 3: dft_axes = (-2,-1)
	elif X.shape[-1] == 3: dft_axes = (-3,-2)
	else: print('Error, input should be hwc or chw'); quit()


	dft = np.log(eps+np.abs(np.fft.fftshift(np.fft.fft2(X.astype('float'),axes=dft_axes))))
	dft -= dft.min()
	dft /= dft.max()
	dft = 2*dft - 1
	return dft




if __name__=='__main__':

	import matplotlib.pyplot as plt

	data_dir = '/media/ssd2/mike/gan_data_trimmed/cyclegan/horse2zebra/'

	files = open(data_dir+'rel_file_path.txt').read().split('\n')[2].split('\t')
	files = [data_dir + f for f in files]

	X_list = [image_reader(f) for f in files]

	dft_list = [(normed_dft(X)[:,:,0]+1)/2 for X in X_list]

	fig, axes = plt.subplots(2,2)

	for a, X in zip(axes.reshape(-1),X_list + dft_list):
		a.imshow(X)

	plt.show()



	#print(files)








