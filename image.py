import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random 
import openslide
import skimage.measure
from PIL.ImageOps import invert


mode= 'train'
size = 128
img_id = "001c62abd11fa4b57bf7a6c603a11bb9"
img_dir = "../train_images"


def topk(X, n):
    x = np.zeros(n, dtype=int)
    y = np.zeros(n, dtype=int)
    x_ = np.ravel(X)
    for i in range(n):
        ij = x_.argmax()
        x_[ij] = 0
        x[i], y[i] = np.unravel_index(ij, X.shape)
    return x, y
      
image_path = os.path.join(img_dir, img_id + '.tiff')
image = openslide.OpenSlide(image_path)
w0,h0 = image.level_dimensions[0]
view = (64,64)
thumbnail = invert(image.get_thumbnail(view))  
img = np.array(thumbnail).mean(2)
w1,h1 = thumbnail.size
num =  {32:24}
images = []
debug=True
if debug:
    fig,ax = plt.subplots(1)
    ax.imshow(img)
for level, n in num.items():
    r = 64 // level
    label = skimage.measure.block_reduce(img, (r,r), np.mean)
    xs,ys = topk(label,30)
    ll=list(range(30))
    if mode == 'train':random.shuffle(ll)
    ll = ll[:n]
    pts = [(x,y) for x,y in zip(xs[ll],ys[ll])]
    for x,y in pts:
        s0 = max(w0,h0)
        l = image.get_best_level_for_downsample(s0//(level*size))
        s = max(image.level_dimensions[l])
        ix,iy = x*s0//level , y*s0//level
        crop_size = s//level
        if mode == 'train':
            t = s0//level//8
            ix += random.randrange(-t, t)
            iy += random.randrange(-t, t)
            crop_size = int(random.uniform(0.8*crop_size,1.2*crop_size))
            
        print((w0,h0),(iy,ix), l, (crop_size,crop_size))
        im = image.read_region((iy,ix), l, (crop_size,crop_size))    
       
        if debug:
            rect = patches.Rectangle((y*r,x*r),r,r,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        im = im.resize((size,size)).convert('RGB')
        images += [im]
plt.savefig("p.png")

for i,im in enumerate(images):
    plt.figure()
    plt.imshow(im)
    plt.savefig(str(i)+".png")