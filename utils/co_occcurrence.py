
import torch
#from 


def co_occur_vert(X):

	h,w,c = X.shape
	for i in range(c):
		X_i = X[:,:,i]
		pairs = torch.stack((X_i[:-1].view(-1),X_i[1:].view(-1)),dim=1)
		print(pairs)


X = torch.randint(0,4,(4,5,1))
print(X[:,:,0])

print(co_occur_vert(X))





