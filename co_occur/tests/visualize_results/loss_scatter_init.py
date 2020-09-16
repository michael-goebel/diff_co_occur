import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from random import shuffle

def read_last_line(fname):
	with open(fname) as f: return [float(i) for i in f.read().split('\n')[0].split(' ')]


data_dir = '/media/ssd2/mike/outputs_3/'



diff_lamb = [data_dir + f'lambda_{l}/' for l in [0.0, 3.0, 10.0]]

print(diff_lamb)

all_files = [glob(d + '*/*/loss.txt') for d in diff_lamb]

for f in all_files: shuffle(f)

labels = [r'$\lambda = 0.0$', r'$\lambda = 3.0$', r'$\lambda = 10.0$']

X = np.array([read_last_line(f) for f in all_files[2][:]])

print(np.mean(X[:,2]))

#for f_list,l in zip(all_files,labels):

#	X = np.array([read_last_line(f) for f in f_list[:200]])

#	plt.plot(X.T[1],X.T[2],'.',label=l)

#	print(np.mean(X[:,2]))

#	print(X)


#print([len(i) for i in all_files])








