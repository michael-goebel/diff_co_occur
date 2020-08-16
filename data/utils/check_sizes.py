from PIL import Image

import os
from glob import glob
from tqdm import tqdm

from collections import Counter

all_files = glob('/media/ssd1/mike/gan_data_trimmed/split_files/*.txt')




#print(all_files)
bad_list = list()

for fname_txt in all_files:

	with open(fname_txt) as f: file_list = f.read().split('\n')

	print(fname_txt)
	#print(len(file_list))
	good_list = list()

	n_bad = 0
	n_spade = 0

	for fname in tqdm(file_list):
		img = Image.open(fname)
		s = img.size
		img.close()

		if min(s) < 256:
			n_bad += 1
			bad_list.append(fname)
		else:
			good_list.append('/'.join(fname.split('/')[5:7]))


	print(len(file_list),n_bad)
	print(Counter(good_list))

#print([f.split('/')[6] for f in bad_list])



