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

from pre_proc import RandomCrop, CenterCrop, JPEGFilter, Resize, hwc2chw

from advertorch.attacks import LinfPGDAttack
from advertorch_examples.utils import bhwc2bchw, bchw2bhwc

def read_txt(fname):
	with open(fname) as f: return f.read().split('\n')


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

model_name = 'resnet18'
method = 'dft'


#data_dir = '../data/cat_test/'
#model_path = '../detection/direct/model_state_dict.h5'

data_dir = '/media/ssd2/mike/gan_data_trimmed/split_files/'


files_real, files_fake = [read_txt(f'{data_dir}adv_train_{t}.txt') for t in ['real','fake']]

model = get_model(model_name,method).to('cuda')

#device = model.device
device = 'cuda'

#model = torch.nn.DataParallel(model)


in_dir = glob(f'../detection/outputs_*/{method}_{model_name}/')[0]
print(in_dir)
#print(model.state_dict().keys())

model.load_state_dict(torch.load(in_dir + 'model.h5'))





model.eval()

adversary = LinfPGDAttack(
	model, eps=1.0, eps_iter=1.0*2/40, nb_iter=40,
	rand_init=True, targeted=False,
	clip_min=0.0, clip_max=255.0
)

#img_orig = np.array(Image.open(f_fake))
#img = bhwc2bchw(img_orig[None,:,:,:]).astype(float)

#img_fake = files_fake[0]

#print(model.device)
#quit()

img_fake = image_reader(files_fake[3].replace('ssd1','ssd2'))

for func in [CenterCrop(256), hwc2chw]: img_fake = func(img_fake)

print(img_fake.shape)


img_torch = torch.Tensor(img_fake).float().unsqueeze(0).to(device)
print(img_torch)

label = torch.Tensor([1]).long().to('cuda')


adv_img = adversary.perturb(img_torch,label)

print(img_torch)
print(adv_img)
print(img_torch - adv_img)

#print(model(img_torch))

print(model(img_torch), model(adv_img), model(torch.round(adv_img))  )


#adv_img_np = bchw2bhwc(adv_img.numpy())[0]
#adv_img_np = ((adv_img_np + 1)*128).astype('uint8')

#fig, axes = plt.subplots(1,3)

#print(img_orig)
#print(adv_img_np)

#axes[0].imshow(img_orig)
#axes[2].imshow(adv_img_np)
#plt.show()





