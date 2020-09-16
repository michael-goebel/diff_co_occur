import sys, os
import torch
from glob import glob
from tqdm import tqdm

sys.path.append('../../utils/')
sys.path.append('../../../utils/')

from my_alg_cband import run_alg, VideoGen, get_losses, round_rand, savefig
from one_d_dist import get_pairs, apply_funcs

from pre_proc import CenterCrop
from image_reader import image_reader, image_writer

gpu = sys.argv[1]
lamb = float(sys.argv[2])

start_ind = int(sys.argv[3])
end_ind = int(sys.argv[4])

start_block = int(sys.argv[5])
end_block = int(sys.argv[6])

tvt_type = sys.argv[7]

ap_list = [{'n_steps': 200, 'sigma': 0.01, 'lamb': lamb},
	   {'n_steps': 50, 'sigma': 0.01, 'lamb': lamb},
	   {'n_steps': 50, 'sigma': 0.00, 'lamb': lamb}]

n_save = 10

old_data_dir = '/media/ssd1/mike/gan_data_trimmed/'
data_dir = '/media/ssd2/mike/gan_data_trimmed/'

output_dir = f'outputs_3/lambda_{lamb}/{tvt_type}'



if not os.path.exists(output_dir): os.makedirs(output_dir)

ht_params = {'n_layers': 9, 'n_bins': 256, 'interp': 'raised cos'}
optim_params = {'lr': 0.01, 'momentum': 0.9}
pre_proc_funcs = [CenterCrop(256),]

os.environ['CUDA_VISIBLE_DEVICES']=gpu

with open(data_dir + f'split_files/adv_{tvt_type}_real.txt') as f: real_list = f.read().split('\n')
with open(data_dir + f'split_files/adv_{tvt_type}_fake.txt') as f: fake_list = f.read().split('\n')

fake_list = fake_list[start_block:end_block]
real_list = real_list[start_block:end_block]

fake_list = [f.replace(old_data_dir,data_dir) for f in fake_list]
real_list = [f.replace(old_data_dir,data_dir) for f in real_list]


fake_real_tuples = get_pairs(fake_list, real_list, pre_proc_funcs)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


for fake_img, real_img in tqdm(fake_real_tuples[start_ind:end_ind]):

	out_dir_i = os.path.join(output_dir,fake_img.split('/')[-1].split('.')[0])
	if not os.path.exists(out_dir_i): os.mkdir(out_dir_i)

	I1_np, I2_np = [image_reader(f).astype('single') for f in [fake_img,real_img]]
	I1_np, I2_np = [apply_funcs(I,pre_proc_funcs) for I in [I1_np,I2_np]]

	I1, I2 = [torch.tensor(i).to(device) for i in [I1_np,I2_np]]
	I1_orig = I1.clone().detach()
	I_list = [I1_orig,I1,I2]

	loss = None
	hist = list()
	
	index = 0
	for alg_params in ap_list:
		for i, loss in enumerate(run_alg(I1_orig,I1,I2,ht_params,optim_params,**alg_params)):
			index += 1
			if i % n_save == 0:
				img_l1, cc_l1 = get_losses(I1_orig,I1,I2,ht_params)
				hist.append([float(val) for val in [loss,img_l1,cc_l1]])
		I1.data = round_rand(I1)

	img_l1, cc_l1 = get_losses(I1_orig,I1,I2,ht_params)
	hist.append([float(val) for val in [loss,img_l1,cc_l1]])

	hist_str = '\n'.join([' '.join([str(val) for val in row]) for row in hist])
	with open(os.path.join(out_dir_i,'loss.txt'),'w') as f: f.write(hist_str)
	image_writer(os.path.join(out_dir_i,'output.png'),I1.detach().cpu().numpy())
	with open(os.path.join(out_dir_i,'files.txt'),'w') as f: f.write(f'{fake_img}\n{real_img}')





