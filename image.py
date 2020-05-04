import matplotlib.pyplot as plt
from train import extract_images

mode= 'train'
size = 128
img_id = "001c62abd11fa4b57bf7a6c603a11bb9"
image_path = "../train_images"
images = extract_images("001c62abd11fa4b57bf7a6c603a11bb9",image_path,128,"train")
for i,im in enumerate(images):
    plt.figure()
    plt.savefig(str(i)+".png")