import numpy as np

from glob import glob
import os

all_dirs = glob('outputs_2/*')

print(all_dirs)

import matplotlib.pyplot as plt

def get_last_losses(fname):
	with open(fname) as f: line = f.readlines()[-1]
	return [float(i) for i in line.split(' ')[1:]]
	


for lamb_dir in all_dirs:

	all_files = glob(os.path.join(lamb_dir,'*/loss.txt'))
	
	losses = np.array([get_last_losses(f) for f in all_files]).T

	label = lamb_dir.split('/')[-1].replace('_', ' = ')

	plt.scatter(losses[0], losses[1], label=label)

	print(lamb_dir)
	print(np.mean(losses,1))


plt.xlabel('Image Mean Abs Difference')
plt.ylabel('Co-Occurrence Mean Abs Difference')

plt.legend()
plt.show()

	



