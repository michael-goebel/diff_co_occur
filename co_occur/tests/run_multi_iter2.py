import sys, os
import torch
from glob import glob

sys.path.append('../utils/')
from my_alg import run_alg, VideoGen, get_losses, round_rand
from one_d_dist import get_pairs 

sys.path.append('../../utils/')
from pre_proc import CenterCrop
from image_reader import image_reader


ap_list = [{'n_steps': 10, 'sigma': 0.01, 'lamb': 0},
	   {'n_steps': 10, 'sigma': 0.01, 'lamb': 0},
	   {'n_steps': 10, 'sigma': 0.0, 'lamb': 0}]

n_save = 5
output_dir = 'outputs2/'

ht_params = {'n_layers': 9, 'n_bins': 256, 'interp': 'raised cos'}
optim_params = {'lr': 0.01, 'momentum': 0.9}
pre_proc_funcs = [CenterCrop(256),]

os.environ['CUDA_VISIBLE_DEVICES']='2'

real_list = glob('/home/mike/spring_2020/gan_images/real/*')
fake_list = glob('/home/mike/spring_2020/gan_images/fake/*')

fake_real_tuples = get_pairs(fake_list, real_list, pre_proc_funcs)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')



for fake_img, real_img in fake_real_tuples:

	out_dir_i = os.path.join(output_dir,fake_img.split('/')[-1].split('.')[0])

	video_obj = VideoGen(out_dir_i,ap_list)

	I1_np, I2_np = [image_reader(f).astype('single') for f in [fake_img,real_img]]
	I1, I2 = [torch.tensor(i).to(device) for i in [I1_np,I2_np]]
	I1_orig = I1.clone().detach()
	I_list = [I1_orig,I1,I2]
	loss = None

	
	index = 0
	for alg_params in ap_list:
		for i, loss in enumerate(run_alg(I1_orig,I1,I2,ht_params,optim_params,**alg_params)):
			index += 1
			if i % n_save == 0: video_obj.add_fig(I1_orig,I1,I2,ht_params,index)
			print(index, get_losses(I1_orig,I1,I2,ht_params))

		I1.data = round_rand(I1)
		print(index, get_losses(I1_orig,I1,I2,ht_params))

	video_obj.save()


#print_err()

#from hist import RaisedCos, hist_tree, hist_loss


#out_dir = f'outputs/seed_{seed}'
#if not os.path.exists(out_dir): os.makedirs(out_dir)
 
#for i, _ in enumerate(run_alg(I1,I1_orig,I2,ht_params,optim_params,**alg_params)):
#	if i % n_save == 0:
#		title = f'Step {i} of {alg_params["n_steps"]}'
#		savefig(*I_list,title,ht_params,os.path.join(out_dir,f'{i:03d}.png'))

#os.chdir(out_dir)
#subprocess.call(['convert', '-delay', '30', '*.png', 'out.mp4'])


