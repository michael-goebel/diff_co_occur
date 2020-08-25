import sys, os
import torch, torchvision

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append('../utils/')
sys.path.append('../detection/utils/')
from train import get_model

from pre_proc import CenterCrop
from image_reader import image_reader, image_writer

from pre_proc import RandomCrop, CenterCrop, JPEGFilter, Resize, hwc2chw, chw2hwc

from advertorch.attacks import LinfPGDAttack
from advertorch_examples.utils import bhwc2bchw, bchw2bhwc

from tqdm import trange
from glob import glob

def read_txt(fname):
	with open(fname) as f: return f.read().split('\n')


#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

method = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

model_name = 'resnet18'
#method = 'dft'
bs = 32


#tvt = 'train'

data_dir = '/media/ssd2/mike/gan_data_trimmed/split_files/'


#for tvt in ['train', 'val', 'test']:
for tvt in ['test']:


	out_dir = f'outputs_3/{method}_{model_name}_pretrained/{tvt}/'
	files_fake = read_txt(f'{data_dir}adv_{tvt}_fake.txt')
	model = get_model(model_name,method).to('cuda')
	device = 'cuda'

	print(len(files_fake))

	in_dir = glob(f'../detection/outputs_*/{method}_{model_name}_pretrained/')[0]
	model.load_state_dict(torch.load(in_dir + 'model.h5'))

	model.eval()

	adversary = LinfPGDAttack(
		model, eps=1.0, eps_iter=1.0*2/40, nb_iter=40,
		rand_init=True, targeted=False,
		clip_min=0.0, clip_max=255.0
	)

	N = len(files_fake)

	def load_img(fname):
		img = image_reader(fname.replace('ssd1','ssd2'))
		for func in [CenterCrop(256), hwc2chw]: img = func(img)
		return img


	for i in trange(int(np.ceil(N/bs))):

#	for i in [int(np.ceil(N/bs))-1,]:

		inds = i*bs + np.arange(bs)
		batch_files = files_fake[i*bs:(i+1)*bs]

		batch_np = np.stack([load_img(f) for f in batch_files])

		img_torch = torch.Tensor(batch_np).float().to(device)
		#label = torch.ones(bs).long().to(device)
		label = torch.ones(len(batch_files)).long().to(device)

		adv_img = adversary.perturb(img_torch,label)
		adv_img = torch.round(adv_img)

		adv_np = adv_img.detach().cpu().numpy()

		for j, fname, img in zip(inds, batch_files, adv_np):
			subdir = out_dir + f'{j:05d}/'
			if not os.path.exists(subdir): os.makedirs(subdir)
			image_writer(subdir + 'output.png', chw2hwc(img))
			with open(subdir + 'input_file.txt','w+') as f: f.write(fname)


