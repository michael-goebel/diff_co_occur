import sys, os, argparse
import torch
from tqdm import tqdm

from utils.image.pre_proc import CenterCrop
from utils.image.image_io import image_reader, image_writer

from utils.co_occur.one_d_dist import get_pairs, apply_funcs
from utils.co_occur.gbco import run_alg, get_losses, round_rand

tvt = ['train','val','test']

parser = argparse.ArgumentParser(description="Generate gray-box co-occurrence attack samples")
parser.add_argument('--gpu_id',help='Index of gpu to use.')
parser.add_argument('--data_split',help=f'Which data split to use. Should be one of {tvt}.')
parser.add_argument('--lamb',type=float,help=f'Lambda constant, discussed in the paper.')
parser.add_argument('--block',help='Comma-separated start and \
		stop indices for block, such as \"1000,2000\".')
parser.add_argument('--inds',help='Comma-separated start and stop \
		indices within each block. For example, if your block is \"1000,2000\", then you can \
		split this across 2 GPUs by setting \"--inds\" to \"0,500\" for one process, and \
		\"500,1000\" for the other.')
parser.add_argument('--reverse',action='store_true',help='Reverse the attack, uses real as \
		the source and GAN as the target.')
parser.add_argument('--cband',action='store_true',help='Runs algorithm using cross-band \
		co-occurrence pairs.')
args = parser.parse_args()


print(args.reverse, args.cband)


start_ind, end_ind = [int(i) for i in args.inds.split(',')]
start_block, end_block = [int(i) for i in args.block.split(',')]
tvt_type = args.data_split

ap_list = [{'n_steps': 200, 'sigma': 0.01, 'lamb': args.lamb, 'cband': args.cband},
	   {'n_steps': 50, 'sigma': 0.01, 'lamb': args.lamb, 'cband': args.cband},
	   {'n_steps': 50, 'sigma': 0.00, 'lamb': args.lamb, 'cband': args.cband}]

n_save = 10

in_dir = '../data/original/'

extras_list = [l for l,b in [['cband',args.cband],['reverse',args.reverse]] if b]

extras_list += [f'lambda_{args.lamb:0.1f}',]

out_dir = f'../data/adversarial/co_occur_gb/{"_".join(extras_list)}/{tvt_type}/'
print(out_dir)

if not os.path.exists(out_dir): os.makedirs(out_dir)

ht_params = {'n_layers': 9, 'n_bins': 256, 'interp': 'raised cos'}
optim_params = {'lr': 0.01, 'momentum': 0.9}
pre_proc_funcs = [CenterCrop(256),]

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

with open(in_dir + f'data_splits/adv_{tvt_type}_real.txt') as f: real_list = f.read().split('\n')
with open(in_dir + f'data_splits/adv_{tvt_type}_fake.txt') as f: fake_list = f.read().split('\n')

fake_list = [in_dir + f for f in fake_list[start_block:end_block]]
real_list = [in_dir + f for f in real_list[start_block:end_block]]

fake_real_tuples = get_pairs(fake_list, real_list, pre_proc_funcs)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

for i, (fake_img,real_img) in enumerate(tqdm(fake_real_tuples[start_ind:end_ind])):

	out_dir_i = out_dir + '{start_block+start_ind+i}/'

	if not os.path.exists(out_dir_i): os.mkdir(out_dir_i)

	f_list = [real_img,fake_img] if args.reverse else [fake_img,real_img]

	I1_np, I2_np = [image_reader(f).astype('single') for f in [fake_img,real_img]]
	I1_np, I2_np = [apply_funcs(I,pre_proc_funcs) for I in [I1_np,I2_np]]

	I1, I2 = [torch.tensor(i).to(device) for i in [I1_np,I2_np]]
	I1_orig = I1.clone().detach()
	I_list = [I1_orig,I1,I2]

	loss = None
	hist = list()
	
	for alg_params in ap_list:
		for i, loss in enumerate(run_alg(I1_orig,I1,I2,ht_params,optim_params,**alg_params)):
			if i % n_save == 0:
				img_l1, cc_l1 = get_losses(I1_orig,I1,I2,ht_params)
				hist.append([float(val) for val in [loss,img_l1,cc_l1]])
		I1.data = round_rand(I1)

	img_l1, cc_l1 = get_losses(I1_orig,I1,I2,ht_params)
	hist.append([float(val) for val in [loss,img_l1,cc_l1]])

	image_writer(os.path.join(out_dir_i,'output.png'),I1.detach().cpu().numpy())
	hist_str = '\n'.join([' '.join([str(val) for val in row]) for row in hist])
	with open(os.path.join(out_dir_i,'loss.txt'),'w') as f: f.write(hist_str)
	with open(os.path.join(out_dir_i,'files.txt'),'w') as f: f.write(f'{fake_img}\n{real_img}')



