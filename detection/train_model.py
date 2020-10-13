import torch
import os, sys, argparse

from utils.detection.data_gen import get_data_gen 
from utils.image.pre_proc import CenterCrop, hwc2chw
from utils.detection.run_model import train_and_val, get_opt_and_loss
from utils.detection.load_model import get_model

all_methods = ['co_occur', 'cband_co_occur', 'dft', 'direct']

parser = argparse.ArgumentParser(description='Train a GAN detector')
parser.add_argument('--n_epochs',default=16,help='Number of epochs')
parser.add_argument('--gpu_id',default=str(),help='Index of the GPU to use.')
parser.add_argument('--n_cpu',default=1,help='Number of CPU processes for data generator')
parser.add_argument('--random_init',action='store_true',help='If true, CNN will start at random \
		initialization. Otherwise, weights will be initialized with ImageNet values.')
parser.add_argument('--jpeg_q',default=None,help='Leave as \"None\" for no JPEG compression, \
		or specify a quality factor 0-100.')
parser.add_argument('--method',help=f'Pre-processing step before CNN. One of {all_methods}.')
parser.add_argument('--model',help=f'CNN name. See utils/detection/load_model.py for all \
		available models')
parser.add_argument('--adv_data',default='none',help=f'Denotes the adversarial files to include for training. \
		For example, none, co_3.0, all_dft, or all_adv')
args = parser.parse_args()

if args.gpu_id in ['-1','none','None']: gpu = str()
else: gpu = args.gpu_id

os.environ['CUDA_VISIBLE_DEVICES'] = gpu

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pre_proc_funcs = [CenterCrop(256), hwc2chw]
if args.jpeg_q is not None: pre_proc_funcs = [JPEGFilter(args.jpeg_q),] + pre_proc_funcs


bs_real = 20 if args.adv_data == 'all_adv' else 16

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

tr_dg = get_data_gen(args.adv_data,'train',pre_proc_funcs,args.n_cpu)
va_dg = get_data_gen(args.adv_data,'val',pre_proc_funcs,args.n_cpu)

model = get_model(args.model,args.method,not(args.random_init)).to(device)
model_parallel = model if len(gpu.split(','))==1 else torch.nn.DataParallel(model)
optimizer, loss_func = get_opt_and_loss(model_parallel)

out_dir = f'../models/{args.method}_{args.model}_ADV_DATA_{args.adv_data}_JPEG_{str(args.jpeg_q).lower()}_' \
		+ f'RAND_INIT_{str(args.random_init).lower()}/'



print(out_dir)

if not os.path.exists(out_dir): os.makedirs(out_dir)

train_and_val(out_dir,model,model_parallel,optimizer,loss_func,tr_dg,va_dg,args.n_epochs,show_pbar=True)



