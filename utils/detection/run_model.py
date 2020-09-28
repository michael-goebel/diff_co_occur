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







