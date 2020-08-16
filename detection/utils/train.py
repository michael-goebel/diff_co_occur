from tqdm import tqdm
import torch, torchvision
import numpy as np


all_model_names = ['vgg16', 'resnet18', 'resnet152', 'resnext50', 'inception_v3']

def get_model(name):
	assert name in all_model_names, print(f'Model {name} not in accepted models: {all_model_names}')

	if name == 'vgg16':
		model = torchvision.models.vgg16(pretrained=False)
		model.fc = torch.nn.Linear(4096,2)
		return model

	if name == 'resnet18':
		model = torchvision.models.resnet18(pretrained=False)
		model.fc = torch.nn.Linear(512,2)
		return model

	if name == 'resnet152':
		model = torchvision.models.resnet152(pretrained=False)
		model.fc = torch.nn.Linear(2048,2)
		return model

	if name == 'resnext50':
		model = torchvision.models.resnext50_32x4d(pretrained=False)
		model.fc = torch.nn.Linear(2048,2)
		return model
	
	if name == 'inception_v3':
		model = torchvision.models.inception_v3(pretrained=False)
		model.fc = torch.nn.Linear(2048,2)
		return model		




def get_opt_and_loss(model): return (torch.optim.Adam(model.parameters()),torch.nn.CrossEntropyLoss())

def read_txt(fname):
	with open(fname) as f: return f.read().split('\n')


def run_model(model,optimizer,loss_func,data_gen,train,show_pbar=True,ind_start=0,n_batch=None):

	if train: model.train()
	else: model.eval()
	run_loss, run_acc = 0, 0

	nc = data_gen.n_class
	ng = data_gen.n_group	

	conf = np.zeros((nc,ng))
	meter = tqdm if show_pbar else iter

	iterator = meter(data_gen)
	print(iterator)
	data_gen.i = ind_start
	data_gen.n_batch = max_n_batch

	for x,y,g in iterator:

		print(data_gen.i)
		x_torch = torch.tensor(x).float()
		y_torch = torch.tensor(y)

		if train: optimizer.zero_grad()
		y_pred = model(x_torch)

		loss = loss_func(y_pred,y_torch)
		if train: loss.backward(); optimizer.step()

		y_pred_ind = y_pred.detach().cpu().numpy().argmax(1)
		conf += np.bincount(ng*y_pred_ind + g, minlength=ng*nc).reshape((nc,ng))

		run_loss += float(loss)*y_torch.shape[0]
		run_acc += float(torch.sum( (y_torch == y_pred.argmax(1)).float()))

	if train:
		n_samp = len(data_gen) * sum(data_gen.npb)
		return run_loss/n_samp, run_acc/n_samp, conf/data_gen.npb[None,:]/len(data_gen)
	else:
		n_samp = len(data_gen.files)
		n_per_class = np.array([len(f) for f in data_gen.files_lists])
		return run_loss/n_samp, run_acc/n_samp, conf/n_per_class[None,:]










