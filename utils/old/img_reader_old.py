from PIL import Image
#import webp
import numpy as np

def image_reader(f):
	try:
		return np.array(Image.open(f).convert('RGB'))
	except:
		print(f'error with file: {f}'); quit()


#	try:
#	        if f.endswith('.webp'): return np.array(webp.load_image(f,'RGB'))
#        	else: return np.array(Image.open(f).convert('RGB'))
#	except:
#		print(f); quit()

def image_writer(name,X):
	if X.max() > 1:
		X_out = np.round(X).astype('uint8')
		Image.fromarray(X_out).save(name)

