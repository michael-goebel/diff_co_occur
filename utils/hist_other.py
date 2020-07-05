
def hist_tree(X,H,n_bins,n_layers,interp):
        return [H.apply(X/(2**i),n_bins//(2**i)+1,interp) for i in range(n_layers)]

def hist_loss(H1,H2): return sum([(2**i)*torch.sum(torch.abs(h1-h2)) for i,(h1,h2) in enumerate(zip(H1,H2))])




