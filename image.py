import os
import numpy as np
import matplotlib.pyplot as plt
import random 
import openslide
import skimage.measure
import PIL
from PIL.ImageOps import invert

def topk(X, n):
    x = np.zeros(n, dtype=int)
    y = np.zeros(n, dtype=int)
    x_ = np.ravel(X)
    for i in range(n):
        ij = x_.argmax()
        x_[ij] = 0
        x[i], y[i] = np.unravel_index(ij, X.shape)
    return x, y

mode= 'train'
size = 128
img_id = "001c62abd11fa4b57bf7a6c603a11bb9"
image_path = os.path.join("../train_images", img_id + '.tiff')
image = openslide.OpenSlide(image_path)
w0,h0 = image.level_dimensions[0]
thumbnail = invert(image.get_thumbnail((size,size)))  
img = np.array(thumbnail).mean(2)
w1,h1 = thumbnail.size
im = PIL.Image.new('RGB',(size,size))
im.paste(thumbnail, (random.randrange(size+1-w1), random.randrange(size+1-h1)))
num =  {16:4, 32:4, 64:4, 128:4}
images = [im]

for level, n in num.items():
    r = size // level
    label = skimage.measure.block_reduce(img, (r,r), np.mean)
    plt.figure()
    plt.imshow(label*255)
    plt.savefig("l"+str(level)+".png")
    xs,ys = topk(label,12)
    ll=list(range(12))
    random.shuffle(ll)
    ll = ll[:n]
    pts = [(x,y) for x,y in zip(xs[ll],ys[ll])]
    print(pts)
    for x,y in pts:
        s0 = max(w0,h0)
        l = image.get_best_level_for_downsample(s0//(level*size))
        s = max(image.level_dimensions[l])
        ix,iy = x*s0//level , y*s0//level
        if mode == 'train':
            t = s0//level//2 - 1
            ix += random.randrange(-t, t)
            iy += random.randrange(-t, t)
        im = image.read_region((iy,ix), l, (s//level,s//level))        
        im = invert(im.resize((size,size)).convert('RGB'))
        images += [im]

    for i,im in enumerate(images):
        plt.figure()
        plt.imshow(np.array(im))
        plt.savefig(str(i)+".png")
