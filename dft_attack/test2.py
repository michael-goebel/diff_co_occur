
import numpy as np
from glob import glob

import sys, os
sys.path.append('../utils/')
from pre_proc import CenterCrop
from image_reader import image_reader, image_writer
from random import shuffle
from tqdm import tqdm


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

all_lists = glob(data_dir + 'adv*real.txt')
groups = [f.split('/')[-1].split('_')[1] for f in all_lists]

f_pairs = [[read_txt(fi) for fi in [f, f.replace('real.txt','fake.txt')]] for f in all_lists]

out_dir = 'outputs_4/'

n_max = 10

def read_and_proc(fname): return CenterCrop(256)(image_reader(fname.replace('ssd1','ssd2')))


for group, (files_real, files_fake) in zip(groups,f_pairs):

	shuffle(files_real)
	shuffle(files_fake)

	for i, (f_real, f_fake) in enumerate(zip(tqdm(files_real), files_fake)):

		img_real = read_and_proc(f_real)
		img_fake = read_and_proc(f_fake)

		for lamb in [0.03, 0.01, 0.003]:

			this_dir = out_dir + f'{group}/lambda_{lamb}/{i:05d}/'
			if not os.path.exists(this_dir): os.makedirs(this_dir)
			img_adv = attack(img_real, img_fake, lamb)
			image_writer(this_dir + 'output.png', img_adv)
			with open(this_dir + 'files.txt','w+') as f: f.write(f'{img_fake}\n{img_real}')

#			print(this_dir)
			


#	print(len(files_real), len(files_fake))





#files_real = read_txt(data_dir + 'adv_train_real.txt')
#files_fake = read_txt(data_dir + 'adv_train_fake.txt')

#img_real = image_reader(files_real[1].replace('ssd1','ssd2'))
#img_fake = image_reader(files_fake[1].replace('ssd1','ssd2'))

#img_adv = attack(img_real, img_fake, 0.003)




