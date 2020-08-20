from tqdm import tqdm
import torch, torchvision, torch.nn as nn
import numpy as np
import os
from torch_pre_proc import CoOccurWithNorm, DFTWithNorm, ImageNetNorm



all_model_names = ['vgg16', 'resnet18', 'resnet50', 'resnet101', 'resnet152',
			'resnext50', 'inception_v3', 'mobilenet', 'squeezenet']


def get_model(name,method,pretrained=True):

	print("pretrained:", pretrained)
	if name == 'vgg16':
		model = torchvision.models.vgg16(pretrained=pretrained)
		model.classifier[6] = torch.nn.Linear(4096,2)

	if name == 'resnet18':
		model = torchvision.models.resnet18(pretrained=pretrained)
		model.fc = torch.nn.Linear(512,2)

	if name == 'resnet50':
		model = torchvision.models.resnet50(pretrained=pretrained)
		model.fc = torch.nn.Linear(2048,2)

	if name == 'resnet101':
		model = torchvision.models.resnet101(pretrained=pretrained)
		model.fc = torch.nn.Linear(2048,2)

	if name == 'resnet152':
		model = torchvision.models.resnet152(pretrained=pretrained)
		model.fc = torch.nn.Linear(2048,2)

	if name == 'resnext50':
		model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
		model.fc = torch.nn.Linear(2048,2)
	
	if name == 'inception_v3':
		model = torchvision.models.inception_v3(pretrained=pretrained)
		model.aux_logits = False
		model.fc = torch.nn.Linear(2048,2)

	if name == 'mobilenet':
		model = torchvision.models.mobilenet_v2(pretrained=pretrained)
		model.classifier[1] = torch.nn.Linear(1280,2)

	if name == 'squeezenet':
		model = torchvision.models.squeezenet1_1(pretrained=pretrained)
		model.classifier[1] = torch.nn.Conv2d(512,2,kernel_size=(1,1), stride=(1,1))



	if method == 'co_occur': pre_proc = CoOccurWithNorm()
	if method == 'dft': pre_proc = DFTWithNorm()
	if method == 'direct': pre_proc = ImageNetNorm()

	return nn.Sequential(pre_proc,model)



def get_opt_and_loss(model): return (torch.optim.Adam(model.parameters()),torch.nn.CrossEntropyLoss())

def read_txt(fname):
	with open(fname) as f: return f.read().split('\n')

def np2csv(X):
	assert len(X.shape) <= 2, print('Input should be a 2D array')
	return '\n'.join([','.join([str(i) for i in l]) for l in X])

# model2save: original model, without parallelization, model: with parallelization
def train_and_val(out_dir,model2save,model,optimizer,loss_func,tr_dg,va_dg,n_val_list,show_pbar=True):

	best_loss = float('inf')
	hist_file = os.path.join(out_dir,'hist.txt')
	with open(hist_file,'w+') as f: f.write(str())
	model_info = str()

	for i, n_val in enumerate(n_val_list):
	
		tr_dg.shuffle()
		tdg_inds = np.linspace(0,tr_dg.L,n_val+1).astype('int')
		for j, (ind_start, n_batch) in enumerate(zip(tdg_inds[:-1],np.diff(tdg_inds))):
			tr_loss, tr_acc, tr_conf = run_model(model,optimizer,loss_func,tr_dg,True,show_pbar,ind_start,n_batch)
			va_loss, va_acc, va_conf = run_model(model,optimizer,loss_func,va_dg,False,show_pbar)

			this_str = f'epoch_{i+1}_pt_{j+1}_of_{n_val}'

			hist_str = f'{this_str}\nTrain:\nLoss: {tr_loss}\nAcc: {tr_acc}\n{np2csv(tr_conf)}\n\nVal:\nLoss: {va_loss}\nAcc: {va_acc}\n{np2csv(va_conf)}\n\n'

			with open(hist_file,'a') as f: f.write(hist_str)			
			print(hist_str)

			if va_loss < best_loss:
				print('saving model')
				torch.save(model2save.state_dict(),os.path.join(out_dir,f'model.h5'))
				best_loss = va_loss
				model_info = this_str

	with open(os.path.join(out_dir,'model_info.txt'),'w+') as f: f.write(model_info)


def run_model(model,optimizer,loss_func,data_gen,train,show_pbar=True,ind_start=0,n_batch=None):

	if train: model.train()
	else: model.eval()
	run_loss, run_acc = 0, 0

	nc = data_gen.n_class
	ng = data_gen.n_group	

	conf = np.zeros((nc,ng))
	meter = tqdm if show_pbar else iter

	data_gen.i = ind_start
	if n_batch is not None: data_gen.n_batch = n_batch	

	iterator = meter(data_gen)
	device = next(model.parameters()).device

	for x,y,g in iterator:

		x_torch = torch.tensor(x).float().to(device)
		y_torch = torch.tensor(y).to(device)

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










