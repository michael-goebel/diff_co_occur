import torch
import itertools
import numpy as np
from hist import hist_tree, hist_loss
from tqdm import trange
import matplotlib.pyplot as plt

def img2pairs(X): return [torch.stack((X[:-1,:,i].view(-1),X[1:,:,i].view(-1)),dim=1) for i in range(X.shape[-1])]



def run_alg(I1,I1_orig,I2,ht_params,optim_params,n_steps,sigma,lamb):

        I1.requires_grad = True
        optimizer = torch.optim.SGD((I1,),**optim_params)
        H2 = [hist_tree(X2_i,**ht_params) for X2_i in img2pairs(I2)]

        for i in trange(n_steps):
                optimizer.zero_grad()
                noise = sigma*torch.randn(I1.shape).type(I1.dtype).to(I1.device)
                I1.data = torch.clamp(I1+noise,0,ht_params['n_bins']-1)
                H1 = [hist_tree(X1_i,**ht_params) for X1_i in img2pairs(I1)]
                loss = sum([hist_loss(H1_i,H2_i) for H1_i, H2_i in zip(H1,H2)])
                loss.backward()
                optimizer.step()
                I1.data = torch.clamp(I1,0,ht_params['n_bins']-1)
                yield loss.detach()


def savefig(I1_orig,I1,I2,title,ht_params,filename):
        col_labels = ['Source', 'Solution', 'Target']
        row_labels = ['Image', 'Red', 'Green', 'Blue']
        X_list = [I.detach() for I in [I1_orig, I1, I2]]
        hist_params = ht_params.copy()
        hist_params['n_layers'] = 1

        C_list = [[hist_tree(X_i,**hist_params)[0] for X_i in img2pairs(X)] for X in X_list]

        img_rmse = torch.sqrt(torch.mean((X_list[0]-X_list[1])**2))
        cc_rmse = torch.sqrt(torch.mean(torch.stack([C_list[1][i]-C_list[2][i] for i in range(3)])**2))

        fig,axes = plt.subplots(4,3)

        for a, X in zip(axes[0],X_list): a.imshow(X.cpu().numpy()/255)
        for a, C in zip(axes[1:].T.reshape(-1),itertools.chain(*C_list)): a.imshow(np.log(1+C.cpu().numpy()))

        for a, l in zip(axes[0],col_labels): a.set_title(l)
        for a, l in zip(axes[:,0],row_labels): a.set_ylabel(l)
        for a in axes.reshape(-1): a.set_xticks([]); a.set_yticks([])

        fig.suptitle(title)
        fig.text(0.02,0.02,f'Image RMSE: {img_rmse:0.3f}\nCo-Occur RMSE: {cc_rmse:0.3f}',fontsize=14)
        fig.set_size_inches(8,8)

        fig.savefig(filename)
        plt.close()




