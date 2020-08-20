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

method = sys.argv[1]
model_name = sys.argv[2]
pt = sys.argv[3]

in_dir = 'outputs_2/' +  '_'.join([method,model_name,pt])


gpu = sys.argv[4]



#method = '_'.join(in_dir.split('/')[1].split('_')[:-1])
#model_name = in_dir.split('/')[1].split('_')[-1]

print(method,model_name)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu


device = 'cuda' if torch.cuda.is_available() else 'cpu'
pre_proc_funcs = [CenterCrop(256), hwc2chw]



bs_list = [16,16]
data_dir = '/media/ssd2/mike/gan_data_trimmed/split_files/'

files_lists = [read_txt('/media/ssd2/mike/gan_data_trimmed/split_files/'+fname) for fname in ['reg_test_real.txt','reg_test_fake.txt']]

adv_dir = '/media/ssd2/mike/outputs_3/lambda_0.0/test/'
files_adv = [adv_dir + f for f in read_txt(adv_dir + 'all_files.txt')]
files_lists.append(files_adv)
print([len(i) for i in files_lists])

te_gen = TestDataGen(files_lists,[0,1,1],sum(bs_list),pre_proc_funcs)


model = get_model(model_name,method).to(device)
model.load_state_dict(torch.load(os.path.join(in_dir,'model.h5')))


optimizer, loss_func = get_opt_and_loss(model)


loss, acc, conf = run_model(model, optimizer, loss_func, te_gen, False)

print(loss, acc, conf)




