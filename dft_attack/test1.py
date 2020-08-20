
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../utils/')
from pre_proc import CenterCrop
from image_reader import image_reader


f_1d = np.array([-1/2,1,-1/2])
f_2d = f_1d[:,None] * f_1d[None,:]
L = 256
f_pad = np.zeros((L,L))
f_pad[:3,:3] = f_2d
f_pad = np.roll(f_pad,(-1,-1),(0,1))
F_filt = np.real(np.fft.fft2(f_pad))


def read_txt(fname):
	with open(fname) as f: return f.read().split('\n')


def attack(img_real, img_fake, lamb):
	F_sq = F_filt**2
	lamb_sq = lamb**2

	F_real = np.fft.fft2(img_real.astype('float'),axes=(0,1),norm='ortho')
	F_fake = np.fft.fft2(img_fake.astype('float'),axes=(0,1),norm='ortho')
	F_out = ( F_sq[:,:,None]*F_real + lamb_sq*F_fake)/( F_sq[:,:,None] + lamb_sq)
	img_out = np.real(np.fft.ifft2(F_out,axes=(0,1),norm='ortho'))
	return np.round(np.clip(img_out,0,255)).astype('uint8')


data_dir = '/media/ssd2/mike/gan_data_trimmed/split_files/'

files_real = read_txt(data_dir + 'adv_train_real.txt')
files_fake = read_txt(data_dir + 'adv_train_fake.txt')

img_real = image_reader(files_real[1].replace('ssd1','ssd2'))
img_fake = image_reader(files_fake[1].replace('ssd1','ssd2'))


img_adv = attack(img_real, img_fake, 0.003)

#print(img_adv)

fig,axes = plt.subplots(2,3)



eps_log = 10**(-6)

for ax,I in zip(axes.T,[img_real, img_fake, img_adv]):
	ax[0].imshow(I)
	dft = np.abs(np.fft.fft2(I,axes=(0,1),norm='ortho'))
	dft_out = np.fft.fftshift(np.log(eps_log + dft)[:,:,0])
	ax[1].imshow(dft_out)


plt.show()




#f_1d = np.array([-1/2,1,-1/2])
#f_2d = f_1d[:,None] * f_1d[None,:]

#L = 256

#f_pad = np.zeros((L,L))
#f_pad[:3,:3] = f_2d
#f_pad = np.roll(f_pad,(-1,-1),(0,1))
#F = np.real(np.fft.fft2(f_pad))

#plt.imshow(F)
#plt.show()



