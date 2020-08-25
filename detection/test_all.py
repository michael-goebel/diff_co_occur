import torch, torchvision
import os, sys
from glob import glob
import numpy as np

sys.path.append('../utils/')
sys.path.append('utils/')


from pre_proc import CenterCrop, hwc2chw
from numpy_co_occur import CoOccur, co_occur_normalize
from numpy_dft import normed_dft
from imagenet_norm import imagenet_norm
from glob import glob
from data_gen import TrainDataGen, TestDataGen
from train import get_model, get_opt_and_loss, run_model, train_and_val, read_txt

def read_txt(fname):
	with open(fname) as f: f_list = f.read().split('\n')
	f_list = [f.replace('/ssd1/','/ssd2/') for f in f_list]
	return f_list



#in_dir = sys.argv[1]




#method = sys.argv[1]
#model_name = sys.argv[2]
#pt = sys.argv[3]

#in_dir = 'outputs_2/' +  '_'.join([method,model_name,pt])


#gpu = sys.argv[4]

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

print(all_groups)

d_paths = [[[d for d in good_dirs if i in d] for i in j] for j in all_groups]
#print(d_paths)

#quit()
#all_groups_flat = [i for row in all_groups for i in row]

scores = -1*np.ones(len(all_methods)*len(all_model_names)*len(all_groups))

files_lists = [read_txt('/media/ssd2/mike/gan_data_trimmed/split_files/'+fname) for fname in ['reg_test_real.txt','reg_test_fake.txt']]
#print(d_paths)
#quit()

for i,d_list in enumerate(flatten_list(d_paths)):



	if len(d_list) == 1:

		in_dir = d_list[0]

		if os.path.exists(in_dir + '/test_results.txt'): continue
		
		method = [m for m in all_methods if m in in_dir][0]
		model_name = [n for n in all_model_names if n in in_dir][0]
		print(method,model_name, in_dir)

		model = get_model(model_name,method).to(device)
		model.load_state_dict(torch.load(os.path.join(in_dir,'model.h5')))

		model = torch.nn.DataParallel(model)

		te_gen = TestDataGen(files_lists,[0,1,1],sum(bs_list),pre_proc_funcs)

		optimizer, loss_func = get_opt_and_loss(model)


		loss, acc, conf = run_model(model, optimizer, loss_func, te_gen, False)
		
		out_str = f'Loss: {loss}\nAcc: {acc}\nPer group: {conf[1,0]}, {conf[1,1]}'

		with open(in_dir + '/test_results.txt', 'w+') as f: f.write(out_str)




