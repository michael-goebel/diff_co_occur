from tqdm import tqdm
import torch
import numpy as np


def get_opt_and_loss(model):
	return torch.optim.Adam(model.parameters()), torch.nn.CrossEntropyLoss()

def np2csv(X):
	assert len(X.shape) <= 2, print('Input should be a 2D array')
	return '\n'.join([','.join([str(i) for i in l]) for l in X])


def run_model(model,optimizer,loss_func,data_gen,train,show_pbar=True,ind_start=0,n_batch=None):

	if train: model.train()
	else: model.eval()
	run_loss, run_acc = 0, 0

	nc = 2
	ng = len(data_gen.batch_sampler.bs_arr)

	conf = np.zeros((nc,ng))
	meter = tqdm if show_pbar else iter

	iterator = meter(data_gen)
	device = next(model.parameters()).device

	for x,y,g in iterator:

		#x_torch = torch.tensor(x).float().to(device)
		#y_torch = torch.tensor(y).to(device)
		x = x.to(device)
		y = y.long().to(device)

		if train: optimizer.zero_grad()
		y_pred = model(x)

		loss = loss_func(y_pred,y)
		if train: loss.backward(); optimizer.step()

		y_pred_ind = y_pred.detach().cpu().numpy().argmax(1)
		conf += np.bincount(ng*y_pred_ind + g.cpu().numpy().astype(int), minlength=ng*nc).reshape((nc,ng))

		run_loss += float(loss)*y.shape[0]
		run_acc += float(torch.sum( (y == y_pred.argmax(1)).float()))

	if train:
		n_samp = len(data_gen) * sum(data_gen.npb)
		return run_loss/n_samp, run_acc/n_samp, conf/data_gen.npb[None,:]/len(data_gen)
	else:
		n_samp = len(data_gen.files)
		n_per_class = np.array([len(f) for f in data_gen.files_lists])
		return run_loss/n_samp, run_acc/n_samp, conf/n_per_class[None,:]



# model2save: original model, without parallelization, model: with parallelization
def train_and_val(out_dir,model2save,model,optimizer,loss_func,tr_dg,va_dg,n_epochs,show_pbar=True):

	best_loss = float('inf')
	hist_file = out_dir + 'hist.txt'
	with open(hist_file,'w+') as f: f.write(str())
	model_info = str()

	for i in range(n_epochs):

		tr_dg.batch_sampler.shuffle()
	
		tr_loss, tr_acc, tr_conf = run_model(model,optimizer,loss_func,tr_dg,True,show_pbar)
		va_loss, va_acc, va_conf = run_model(model,optimizer,loss_func,va_dg,False,show_pbar)

		hist_str = f'Epoch {i}\nTrain:\nLoss: {tr_loss}\nAcc: {tr_acc}\n{np2csv(tr_conf)}\n\nVal:\nLoss: {va_loss}\nAcc: {va_acc}\n{np2csv(va_conf)}\n\n'

		with open(hist_file,'a') as f: f.write(hist_str)
		if va_loss < best_loss:
			print('saving model')
			torch.save(model2save.state_dict(),out_dir+'model.h5')
			best_loss = va_loss
			model_info = this_str
	with open(out_dir+'model_info.txt','w+') as f: f.write(model_info)



