from glob import glob
import numpy as np

from matplotlib import cm

alpha = 0.5

def get_color_str(v):
	c_array = cm.viridis(float(v))[:3]
	c_array = [i*(1-alpha) + alpha for i in c_array]


	c_str = ','.join([f'{i:0.3f}' for i in c_array])
	return '\\cellcolor[rgb]{' + c_str + '}'


#\cellcolor[rgb]{0,1.0,0}


def round(v):
	return f'{float(v):0.3f} {get_color_str(v)}'

def invert(v): return f'{1-float(v)}'

def two_d_list2latex(l):
	L = len(l)
	#out_str = '\\begin{tabular}{|' + 'c|'*L + '}\n'
	out_str = ' \\\\ \n'.join([' & '.join(row) for row in l])
	out_str += ' \\\\ \n\\end{tabular}'
	out_str = out_str.replace('_','\\_')
	return out_str


group_lists = list()

t_1 = ['all_cc', 'all_dft', 'all_pgd', 'all_adv']

t_cc = ['cc_0.0', 'cc_3.0', 'cc_10.0']


#t_list = ['none', 'cc_0.0', 'cc_3.0', 'cc_10.0', 'all_cc', 'all_dft', 'all_pgd', 'all_



group_list = [f'co_occur_resnet18_{t}' for t in ['orig','none'] + t_cc + t_1]

#print(group_list)

group_list += [f'dft_resnet18_{t}' for t in ['orig','none'] + t_1]

group_list += [f'direct_resnet18_{t}' for t in ['orig','none'] + t_1]

group_list += [f'{m}_mobilenet_{t}' for m in ['co_occur','dft','direct'] for t in ['orig','none','all_adv']]
#print(group_list)


#all_files = glob('outputs_4/*/test_results.txt')

all_files = [f'outputs_jpeg/{g}/test_results.txt' for g in group_list]

#print(all_files)

all_data = list()

for fname in all_files:
	with open(fname) as f: all_data.append(f.read().split('\n')[1].split(','))

all_data = [[invert(row[0]),] + row[1:] for row in all_data]

all_data = [[round(i) for i in j] for j in all_data]


with open(all_files[0]) as f: col_labels = f.read().split('\n')[0].split(',')
row_labels = [f.split('/')[1] for f in all_files]


#all_data = [[l,] + row for l,row in zip(row_labels,all_data)]
#all_data = [[' ',] + col_labels] + all_data


latex_list = two_d_list2latex(all_data)

print(latex_list)

print(row_labels)

#model_names = [l.split('_')[0] for l in row_labels]
#print(model_names)

#all_names = ['co_occur', 'direct', 'dft']

method2labels = {'co_occur': 'Co-Occur', 'dft': 'DFT', 'direct': 'Direct'}

names2labels = {'resnet18': 'ResNet18', 'mobilenet': 'MobileNet'}

group2labels = {'orig': 'No Adv*', 'none': 'No Adv', 'cc_0.0': 'GBCO 0.0', 'cc_3.0': 'GB-CO 3.0',\
		'cc_10.0': 'GB-CO 10.0', 'all_cc': 'All GB-CO', 'all_dft': 'All GB-DFT', \
		'all_pgd': 'All PGD', 'all_adv': 'All Adv'}

names = [  [v for k,v in names2labels.items() if k in l][0] for l in row_labels]

names = [str() if n == names[i-1] else n  for i,n in enumerate(names)]

methods = [[v for k,v in method2labels.items() if l.startswith(k)][0] for l in row_labels]

methods = [str() if m == methods[i-1] else m for i,m in enumerate(methods)]


groups = [[v for k,v in group2labels.items() if l.endswith(k)][0] for l in row_labels]


L = len(methods)
r = np.arange(L)

print(names)

inds_names = r[[len(n) > 0 for n in names]]
n_row_names = np.diff(inds_names,append=L)
print(inds_names)

inds_method = r[[len(m) > 0 for m in methods]]
n_rows_method = np.diff(inds_method,append=L)

for ind, n_row in zip(inds_method,n_rows_method):
	methods[ind] = '\\multirow{' + str(n_row) + '}{*}{' + methods[ind] + '}'

for ind, n_row in zip(inds_names,n_row_names):
	names[ind] = '\\multirow{' + str(n_row) + '}{*}{' + names[ind] + '}'

all_data = [' & '.join([n,m,g] + d) + ' \\\\' for n,m,g,d in zip(names,methods,groups,all_data)]

hlines = [str() for _ in range(len(all_data))]

for ind in inds_method: hlines[ind] = '\\cline{2-11} \n'

for ind in inds_names: hlines[ind] = '\\hline \n'

all_data = [h + d for h,d in zip(hlines,all_data)]


print('\n'.join(all_data))



#print(methods)
#print(names)

#for i in range(L):
#	if len(methods[i]) > 0:
#		methods[i] = 




#v = np.diff(r[[len(m) > 0 for m in methods]],append=L)




#print(v)


#print(names)

#print(methods)




#model_names = [



#print(all_data)

#print(two_d_list2latex(all_data))

#latex_list = two_d_list2latex(all_data).split('\n')




#hline_inds = [1,2,2,11,17,23,32]

#for i in hline_inds[::-1]:
#	latex_list.insert(i,'\\hline')

#print('\n'.join(latex_list))


#print(all_data)





