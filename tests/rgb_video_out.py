import sys, os, subprocess
from glob import glob

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append('../utils/')
from hist import RaisedCos, L1Dist, Hist, hist_tree, hist_loss


def img2pairs(X):
	h,w,c = X.shape
	return [torch.stack((X[:-1,:,i].view(-1),X[1:,:,i].view(-1)),dim=1) for i in range(c)]


os.environ['CUDA_VISIBLE_DEVICES']='2'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

real_list = glob('/home/mike/spring_2020/gan_images/real/*')
fake_list = glob('/home/mike/spring_2020/gan_images/fake/*')

seed = int(sys.argv[1])
np.random.seed(seed)

real_img = np.random.choice(real_list)
fake_img = np.random.choice(fake_list)

print(real_img)
print(fake_img)

n_layers = 9
n_steps = 300
n_print = 10
sigma = 0.01
lamb = 0


hist_params = {'n_bins': 256, 'interp': RaisedCos()}
ht_params = {'H': Hist, 'n_layers': 9, **hist_params}
optim_params = {'lr': 0.01, 'momentum': 0.9}
gs_kw = {'cmap':'gray', 'vmin':0, 'vmax':255}

L = 256

I1_np = np.array(Image.open(fake_img),dtype='double')[:L,:L]
I2_np = np.array(Image.open(real_img),dtype='double')[:L,:L]

I1 = torch.tensor(I1_np).double().to(device)
I2 = torch.tensor(I2_np).double().to(device)
I1_orig = I1.clone().detach()
I1.requires_grad = True


X2 = img2pairs(I2)
H2 = [hist_tree(X2_i,**ht_params) for X2_i in X2]

optimizer = torch.optim.SGD((I1,),**optim_params)

out_dir = f'outputs/seed_{seed}'
if not os.path.exists(out_dir): os.makedirs(out_dir)


for i in range(n_steps):

	optimizer.zero_grad()
	noise = sigma*torch.randn(I1.shape).double().to(device)
	I1.data = torch.clamp(I1+noise,0,hist_params['n_bins']-1)
	X1 = img2pairs(I1)
	H1 = [hist_tree(X1_i,**ht_params) for X1_i in X1]
	loss = sum([hist_loss(H1_i,H2_i) for H1_i, H2_i in zip(H1,H2)])
	loss.backward()
	optimizer.step()
	I1.data = torch.clamp(I1,0,hist_params['n_bins']-1)


	if i % n_print == 0:

		col_labels = ['Source', 'Solution', 'Target']
		row_labels = ['Image', 'Red', 'Green', 'Blue']
		X_list = [I.detach().cpu() for I in [I1_orig, I1, I2]]
		C_list = [[Hist.apply(X_i,hist_params['n_bins'],hist_params['interp']) for X_i in img2pairs(X)] for X in X_list]

		img_rmse = torch.sqrt(torch.mean((X_list[0]-X_list[1])**2))
		cc_rmse = torch.sqrt(torch.mean(torch.stack([C_list[1][i]-C_list[2][i] for i in range(3)])**2))

		print(f'{i} of {n_steps}\nLoss: {loss:.2f}\nImg RMSE: {img_rmse:0.3f}\n CC RMSE: {cc_rmse:0.3f}\n')

		fig,axes = plt.subplots(4,3)

		for a, X in zip(axes[0],X_list): a.imshow(X.numpy()/255)
		for ax_col, C_col in zip(axes[1:].T,C_list):
			for a, C in zip(ax_col, C_col):
				a.imshow(np.log(1+C.numpy()))

		for a, l in zip(axes[0],col_labels): a.set_title(l)
		for a, l in zip(axes[:,0],row_labels): a.set_ylabel(l)
		for a in axes.reshape(-1): a.set_xticks([]); a.set_yticks([])

		fig.suptitle(f'Step {i} of {n_steps}')
		fig.text(0.02,0.02,f'Image RMSE: {img_rmse:0.3f}\nCo-Occur RMSE: {cc_rmse:0.3f}',fontsize=14)
		fig.set_size_inches(8,8)


		fig.savefig(os.path.join(out_dir,f'{i//n_print:03d}.png'))
		plt.close()


os.chdir(out_dir)
subprocess.call(['convert', '-delay', '30', '*.png', 'out.mp4'])


