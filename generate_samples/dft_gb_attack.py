import numpy as np
import os
from random import shuffle
from tqdm import tqdm

from utils.image.pre_proc import CenterCrop
from utils.image.image_io import image_reader, image_writer



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

in_dir = '../data/original/'
out_dir = '../data/adversarial/'

#data_dir = '../data/'
tvt = ['train', 'val', 'test']
f_pairs = [[read_txt(in_dir + f'data_splits/adv_{t}_{r}.txt') for r in ['real','fake']] for t in tvt]


def read_and_proc(fname): return CenterCrop(256)(image_reader(fname))


for group, (files_real, files_fake) in zip(tvt,f_pairs):

	shuffle(files_real)
	shuffle(files_fake)

	for i, (f_real, f_fake) in enumerate(zip(tqdm(files_real), files_fake)):

		img_real = read_and_proc(in_dir + f_real)
		img_fake = read_and_proc(in_dir + f_fake)

		for lamb in [0.03, 0.01, 0.003]:

			this_dir = out_dir + f'dft_gb/lambda_{lamb}/{group}/{i:05}/'

			if not os.path.exists(this_dir): os.makedirs(this_dir)
			img_adv = attack(img_real, img_fake, lamb)
			image_writer(this_dir + 'output.png', img_adv)
			with open(this_dir + 'files.txt','w+') as f: f.write(f'{f_fake}\n{f_real}')



