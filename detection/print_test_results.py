import torch, torchvision
import os, sys
from glob import glob
import numpy as np
import matplotlib.cm as cm

sys.path.append('../utils/')
sys.path.append('utils/')


from pre_proc import CenterCrop, hwc2chw
from numpy_co_occur import CoOccur, co_occur_normalize
from numpy_dft import normed_dft
from imagenet_norm import imagenet_norm
from glob import glob
from data_gen import TrainDataGen, TestDataGen
from train import get_model, get_opt_and_loss, run_model, train_and_val, read_txt

alpha = 0.5

def get_color_str(v):
        c_array = cm.viridis(float(v))[:3]
        c_array = [i*(1-alpha) + alpha for i in c_array]


        c_str = ','.join([f'{i:0.3f}' for i in c_array])
        return '\\cellcolor[rgb]{' + c_str + '}'


#\cellcolor[rgb]{0,1.0,0}


def round(v):
        return f'{float(v):0.3f} {get_color_str(v)}'

def two_d_list2latex(l):
        L = len(l)
        out_str = '\\begin{tabular}{|' + 'c|'*L + '}\n'
        out_str += ' \\\\ \n'.join([' & '.join(row) for row in l])
        out_str += ' \\\\ \n\\end{tabular}'
        out_str = out_str.replace('_','\\_')
        return out_str




def read_txt(fname):
	with open(fname) as f: f_list = f.read().split('\n')
	f_list = [f.replace('/ssd1/','/ssd2/') for f in f_list]
	return f_list


os.environ['CUDA_VISIBLE_DEVICES']='1,2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pre_proc_funcs = [CenterCrop(256), hwc2chw]

bs_list = [16,16]



def check_dir(dname):
	with open(dname + '/hist.txt') as f: return 'epoch_16' in f.read()

def flatten_list(l): return [i for j in l for i in j]


all_dirs = glob('outputs_*/*')
valid = [check_dir(d) for d in all_dirs]


good_dirs = [d for d in all_dirs if check_dir(d)]
bad_dirs = [d for d in all_dirs if not check_dir(d)]

good_dirs_trim = [d.split('/')[1] for d in good_dirs]

all_methods = ['co_occur', 'dft', 'direct']

all_model_names = ['vgg16', 'resnet18', 'resnet50', 'resnet101', 'resnet152',
                        'resnext50', 'inception_v3', 'mobilenet', 'squeezenet']

all_pts = ['pretrained', 'randinit']


all_groups = [['_'.join([me,mo,pt]) for me in all_methods for pt in all_pts] for mo in all_model_names]

col_labels = [f'{me}_{pt}' for me in all_methods for pt in all_pts]


d_paths = [[[d for d in good_dirs if i in d] for i in j] for j in all_groups]

scores = -1*np.ones(len(all_methods)*len(all_model_names)*len(all_pts))

files_lists = [read_txt('/media/ssd2/mike/gan_data_trimmed/split_files/'+fname) for fname in ['reg_test_real.txt','reg_test_fake.txt']]

for i,d_list in enumerate(flatten_list(d_paths)):

	if len(d_list) == 1:


		in_dir = d_list[0]
		with open(in_dir + '/test_results.txt') as f: test_txt = f.read()

		scores[i] = float(test_txt.split('Acc: ')[1].split('\n')[0])

		print(scores[i], in_dir)

#		print(test_txt)

scores = scores.reshape((9,6))


def np2list(X): return [[f'{i:0.4f}' for i in j] for j in X]


def line2latex(l): return ' & '.join(l) + ' \\\\'



table = [[str(),] + col_labels,]

#scores_list = np2list(scores)
scores_list = [[round(i) for i in j] for j in scores]


table += [[l,] + v for l,v in zip(all_model_names,scores_list)]


latex_list = two_d_list2latex(table).split('\n')

hline_inds = [1,2,2,11]

for i in hline_inds[::-1]:
	latex_list.insert(i,'\\hline')

print('\n'.join(latex_list))


#latex_lines = [line2latex(l) for l in table]




#hline_inds = [


#latex_table = '\n'.join([line2latex(l) for l in table])
#latex_table = latex_table.replace('_','\\_')

#print(latex_table)

#print(table)





#print(scores)
#print(col_labels)
#print(all_model_names)







#


