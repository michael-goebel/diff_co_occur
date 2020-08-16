from PIL import Image
import webp
import numpy as np

def image_reader(f):
	try:
		img = Image.open(f).convert('RGB')
		size = img.size
		if min(size) < 256:
			new_size = [max(256,i) for i in size]
			img = img.resize(new_size)
		
		return np.array(img)
	except Exception as e:
		print(f'error with file: {f}',e); quit()


#	try:
#	        if f.endswith('.webp'): return np.array(webp.load_image(f,'RGB'))
#        	else: return np.array(Image.open(f).convert('RGB'))
#	except:
#		print(f); quit()

def image_writer(name,X):
	if X.max() > 1:
		X_out = np.round(X).astype('uint8')
		Image.fromarray(X_out).save(name)

