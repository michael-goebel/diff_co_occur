import sys, os
import torch, torchvision

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append('../utils/')
from pre_proc import CenterCrop
from image_reader import image_reader, image_writer

from pre_proc import RandomCrop, CenterCrop, JPEGFilter, Resize

from advertorch.attacks import LinfPGDAttack
from advertorch_examples.utils import bhwc2bchw, bchw2bhwc


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

data_dir = '../data/cat_test/'
model_path = '../detection/direct/model_state_dict.h5'

with open(data_dir + 'tp_adv_files.txt') as f: fake_list = f.read().split('\n')

f_fake = fake_list[2]



model = torchvision.models.resnet.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512,2)

model.load_state_dict(torch.load(model_path))
model.eval()

adversary = LinfPGDAttack(
	model, eps=1.0/255, eps_iter=1.0/255*2/40, nb_iter=40,
	rand_init=True, targeted=False)

img_orig = np.array(Image.open(f_fake))
img = bhwc2bchw(img_orig[None,:,:,:]).astype(float)


img_torch = torch.Tensor(img/128 - 1)

label = torch.Tensor([1]).long()
print(label)


adv_img = adversary.perturb(img_torch,label)
adv_img_np = bchw2bhwc(adv_img.numpy())[0]
adv_img_np = ((adv_img_np + 1)*128).astype('uint8')

fig, axes = plt.subplots(1,3)

print(img_orig)
print(adv_img_np)

axes[0].imshow(img_orig)
axes[2].imshow(adv_img_np)
plt.show()





