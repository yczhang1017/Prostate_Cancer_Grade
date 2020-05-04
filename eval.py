import os
import time
import pandas as pd
import openslide
import numpy as np
import torch

import torch.backends.cudnn as cudnn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import transforms
import argparse
import cv2

parser = argparse.ArgumentParser(
    description='Prostate Cancer Grader')
parser.add_argument('--root', default='..',
                    type=str, help='directory of the data')
parser.add_argument('--workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

parser.add_argument('--dump', default='dump/', type=str,
                    help='Dir to dump results')
parser.add_argument('--size', default=2048, type=int)

args = parser.parse_args()

def getp1(img):
    h,w = img.shape
    for i in range(h): 
        if np.any(img[i,:]<255): break
    for j in range(w): 
        if np.any(img[:,j]<255): break
    return i,j
        
                
    return 0,0
def getp2(img):
    h,w = img.shape
    for i in range(h-1,-1,-1): 
        if np.any(img[i,:]<255): break
    for j in range(w-1,-1,-1): 
        if np.any(img[:,j]<255): break
    return i,j

def crop(image,size):
    view = (256,256)
    im = image.get_thumbnail(view)
    img = np.array(im).mean(2)
    i1,j1 = getp1(img)
    i2,j2 = getp2(img)
    h1,w1=img.shape   #size before crop
    h2,w2 = i2-i1, j2-j1
    r = h2/w2
    hn = size * np.sqrt(r) * h1 // (i2-i1)
    wn = size / np.sqrt(r) * w1 // (j2-j1)
    im = image.get_thumbnail((wn,hn))
    anchor = np.array((j1,i1,j2,i2))
    anchor = anchor * hn // h1
    return im.crop(anchor)

def get_image(img_id, data_dir, size):
    image_path = os.path.join(data_dir, "train_images", img_id + '.tiff')
    image = openslide.OpenSlide(image_path)
    w0, h0 = image.dimensions
    return crop(image,size)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.cuda.set_device(device)
    cudnn.benchmark = True
else:
    device = torch.device("cpu")
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    
class ProstateSeg(Dataset):
    def __init__(self, df, data_dir, size):
        self.df = df
        self.data_dir = data_dir
        self.size = size
       
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx].image_id
        #grade = self.df.iloc[idx].isup_grade
        im = get_image(img_id, self.data_dir, self.size)
        if transform is not None:
            im = transform(im)
        return im
    
    
def main():
    nlabel = 6
    df = pd.read_csv(os.path.join(args.root, "train.csv"))
    dataset = ProstateSeg(df, args.root, 2048)
    loader = DataLoader(dataset,num_workers = 4, pin_memory=True)
    model = models.segmentation.deeplabv3_resnet101(
                pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, nlabel)
    print('Evaluate using {}...'
            .format(args.checkpoint))
    weight_file=os.path.join(args.checkpoint)
    model.load_state_dict(torch.load(weight_file, map_location=lambda storage, loc: storage))
    
    if not os.path.exists(args.dump):
        os.mkdir(args.dump)
    
    model.eval()
    
    with torch.no_grad():
        t0 = time.time()
        for i, inputs in enumerate(loader):
            t1 = time.time()
            imid = df.iloc[i].image_id
            provider =  df.iloc[i].data_provider
            grade = df.iloc[i].isup_grade
            score = df.iloc[i].gleason_score
            inputs = inputs.to(device)   
            output = model(inputs)
            pred = output['out'].argmax(dim=1).detach().cpu()
            
            pp = np.zeros(nlabel)
            npix = np.prod(pred.shape)
            for i in range(nlabel):
                pp[i] = pred.eq(i).sum().item() / npix
            pp /= (pp[1:].sum())
            t2 = time.time()
            print("{:.1f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.1f}|{},{},{},{}s/{}s".format(
                    *pp, grade, score, provider, t2-t1, (t2-t0)/(i+1)))
            torch.save(pred, os.path.join(args.dump, imid))
if __name__ == '__main__':
        main()
