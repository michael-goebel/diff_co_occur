import numpy as np
from random import shuffle
import os


def split_files(fname, splits):

	with open(fname) as f: lines = f.read().split('\n')
	shuffle(lines)

	if len(lines[0].split('\t')) == 2:
		L = len(lines)
		cum_inds = [0,] + [int(np.round(L*i)) for i in np.cumsum(splits)]
		split_lines = [lines[i:j] for (i,j) in zip(cum_inds[:-1],cum_inds[1:])]		
		split_lines = [[i for l in group for i in l.split('\t')] for group in split_lines]
		return split_lines

	else:
		real_lines = [l for l in lines if l.startswith('real')]
		fake_lines = [l for l in lines if l.startswith('fake')]
		L1, L2 = len(real_lines), len(fake_lines)
		assert L1 == L2
		L = L1
		cum_inds = [0,] + [int(np.round(L*i)) for i in np.cumsum(splits)]
		split_lines = [real_lines[i:j] + fake_lines[i:j] for (i,j) in zip(cum_inds[:-1],cum_inds[1:])]
		return split_lines



if __name__=='__main__':


	from glob import glob

	files = glob('/media/ssd1/mike/gan_data_trimmed/*/*/rel_file_path.txt')
	
	tvt_splits = np.array([0.7, 0.15, 0.15])
	reg_adv_splits = np.array([0.5, 0.5])

	tvt_names = ['train', 'val', 'test']
	reg_adv_names = ['reg', 'adv']

	splits = (tvt_splits[None,:] * reg_adv_splits[:,None]).reshape(-1)
	names = [f'{r}_{t}' for r in reg_adv_names for t in tvt_names]

	for f in files:
		out_dir = os.path.dirname(f)

		sub_files = split_files(f,splits)
		for s,n in zip(sub_files,names):
			full_path = os.path.join(out_dir,f'{n}.txt')

			with open(full_path,'w+') as f: f.write('\n'.join(s))

	for n in names:

		files = glob(f'/media/ssd1/mike/gan_data_trimmed/*/*/{n}.txt')
		print(files)
		for gt in ['real','fake']:
			out_list = list()
			for fname in files:
				this_dir = os.path.dirname(fname)
				with open(fname) as f: l = f.read().split('\n')
				l = [os.path.join(this_dir,i) for i in l if i.startswith(gt)]

				out_list += l

			out_path = f'/media/ssd1/mike/gan_data_trimmed/split_files/{n}_{gt}.txt'
			print(out_path)
			shuffle(out_list)
			with open(out_path,'w+') as f: f.write('\n'.join(out_list))


			





