from PIL import Image
import webp
import numpy as np

def image_reader(f):
        if f.endswith('.webp'): return np.array(webp.load_image(f,'RGB'))
        else: return np.array(Image.open(f).convert('RGB'))

