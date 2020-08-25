from glob import glob
import numpy as np


def round(v): return f'{float(v):0.3f}'

def two_d_list2latex(l):
	L = len(l)
	out_str = '\\begin{tabular}{|' + 'c|'*L + '}\n'
	out_str += '\\\\ \n'.join([' & '.join(row) for row in l])
	out_str += '\n \\end{tabular}'
	out_str = out_str.replace('_','\\_')
	return out_str


all_files = glob('outputs_4/*/test_results.txt')

print(all_files)

all_data = list()

for fname in all_files:
	with open(fname) as f: all_data.append(f.read().split('\n')[1].split(','))


all_data = [[round(i) for i in j] for j in all_data]


with open(all_files[0]) as f: col_labels = f.read().split('\n')[0].split(',')
row_labels = [f.split('/')[1] for f in all_files]


all_data = [[l,] + row for l,row in zip(row_labels,all_data)]
all_data = [[' ',] + col_labels] + all_data

print(all_data)

print(two_d_list2latex(all_data))

#print(all_data)





