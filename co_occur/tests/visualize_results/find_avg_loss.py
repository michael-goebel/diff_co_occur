import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from random import shuffle

import sys
sys.path.append('../../utils/')
sys.path.append('../../../utils/')
from image_reader import image_reader
from my_alg import co_occur
from pre_proc import CenterCrop
import torch

def read_last_line(fname):
	with open(fname) as f: return [float(i) for i in f.read().split('\n')[0].split(' ')]


data_dir = '/media/ssd2/mike/outputs_3/lambda_0.0/'

ht_params = {'n_bins':256, 'interp': 'raised cos'}

#diff_lamb = [data_dir + f'lambda_{l}/' for l in [0.0, 3.0, 10.0]]

#print(diff_lamb)

all_files = glob(data_dir + '*/*/files.txt')

shuffle(all_files)

d_sum = 0.0

for i, fname in enumerate(all_files):

	with open(fname) as f: f_fake, f_real = f.read().split('\n')
	img_real = torch.Tensor(CenterCrop(256)(image_reader(f_real.replace('ssd1','ssd2'))))
	img_fake = torch.Tensor(CenterCrop(256)(image_reader(f_fake.replace('ssd1','ssd2'))))
	c_real = co_occur(img_real,ht_params)[0]
	c_fake = co_occur(img_fake,ht_params)[0]

	d = torch.mean(torch.abs(c_real-c_fake))	
	#print(d)

	d_sum += d
	print(d_sum/(i+1), d, i)


	#print(img_real.shape,img_fake.shape)








