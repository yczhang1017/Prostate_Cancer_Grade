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
import pickle
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
parser.add_argument('--size', default=2048, type=int)

args = parser.parse_args()
def get_image(img_id, data_dir, size):
    image_path = os.path.join(data_dir, "train_images", img_id + '.tiff')
    image = openslide.OpenSlide(image_path)
    w0, h0 = image.dimensions
    r = np.sqrt(h0/w0)
    view = (int(size/r), int(size*r))
    im = image.get_thumbnail(view)
    #mm = mask.get_thumbnail(view).getchannel(0)
    return im#,mm

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
    def __init__(self, df, data_dir, size, mode):
        self.df = df
        self.data_dir = data_dir
        self.size = size
        self.mode = mode
       
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx].image_id
        #grade = self.df.iloc[idx].isup_grade
        im,mm = get_image(img_id, self.data_dir, self.size, self.crop, self.mode)
        if transform is not None:
            im = transform(im)
        return im
    
    
def main():
    nlabel = 6
    df = pd.read_csv(os.path.join(args.root, "train.csv"))
    dataset = ProstateSeg(df, args.root, 2048, (600,600), 'val')
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
        for i, inputs in enumerate(loader):
            t0 = time.time()
            imid = df.iloc[i].image_id
            grade = df.iloc[i].isup_grade
            if i==0: print(inputs.shape) 
            inputs = inputs.to(device)   
            output = model(inputs)
            pred = output['out'].argmax(dim=1).detach().cpu()
            
            pp = torch.zeros(nlabel)
            npix = np.prod(pred.shape)
            for i in range(nlabel):
                pp[i] = pred.eq(i).sum() / npix
            print("{}s".format(time.time()-t0), imid, pp ,grade)
            pickle.dump(pred, open(os.path.join(args.dump, imid),"wb"))
            
                