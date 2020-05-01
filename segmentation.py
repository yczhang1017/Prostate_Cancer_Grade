import os
import time
import argparse
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import random 
from random import uniform, randrange
import openslide
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

from models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
 

parser = argparse.ArgumentParser(
    description='Prostate Cancer Grader')
parser.add_argument('--root', default='..',
                    type=str, help='directory of the data')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--epochs', default=24, type=int,
                    help='number of epochs to train')
parser.add_argument('--output_folder', default='save/', type=str,
                    help='Dir to save results')
parser.add_argument('--weight_decay', default=1e-5, type=float,
                    help='Weight decay')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='epoch number to be resumed at')
parser.add_argument('--size', default=2048, type=int)
parser.add_argument('--crop_size', default=640, type=int)
parser.add_argument('--log', default=10, type=int, help='steps to print log')
parser.add_argument('--step', default=8, type=int, help='step to reduce lr')
parser.add_argument('--model', default='fcn_resnest50_ade', type=str)

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.cuda.set_device(device)
    cudnn.benchmark = True
else:
    device = torch.device("cpu")

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)
nlabel = 5

def get_image_mask(img_id, data_dir, size, crop_size, mode='train'):
    image_path = os.path.join(data_dir, "train_images", img_id + '.tiff')
    mask_path = os.path.join(data_dir, "train_label_masks", img_id + '_mask.tiff')
    image = openslide.OpenSlide(image_path)
    mask = openslide.OpenSlide(mask_path)
    if mode == 'train':
        w0,h0 = image.dimensions
        r = h0/w0
        crop_size = int(uniform(0.8,1.25)*crop_size)
        min_size= int(max(crop_size*r,crop_size/r))+1
        size = int(uniform(0.8,1.25)*size)
        size = max(min_size, size)
        w1, h1 = int(size/r), int(size*r)
        x1, y1 = randrange(w1-crop_size+1), randrange(h1-crop_size+1)
        x0, y0 = w0*x1//w1, h0*y1//h1 
        l = image.get_best_level_for_downsample(w0//w1)
        wl, hl = image.level_dimensions[l]
        scrop = crop_size*wl//w1
        im = image.read_region((x0,y0), l, (scrop,scrop)).convert('RGB')
        im = im.resize((crop_size,crop_size))      
        mm = mask.read_region((x0,y0), l, (scrop,scrop)).getchannel(0)
        mm = mm.resize((crop_size,crop_size))   
    else:
        view = (int(size/r), int(size*r))
        im = image.get_thumbnail(view)
        mm = mask.get_thumbnail(view).getchannel(0)
    return im,mm


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    
def radboud_seg(x):
    return x if x<2 else x-1

class ProstateSeg(Dataset):
    def __init__(self, df, data_dir, mode, size, crop_size):
        self.df = df
        self.data_dir = data_dir
        self.size = size
        self.crop_size = crop_size
        self.mode = mode
       
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx].image_id
        im,mm = get_image_mask(img_id, self.data_dir, self.size, self.crop_size, self.mode)
        if self.transform is not None:
            im = self.transform(im)
        
        target = torch.from_numpy(np.array(mm).astype('int64'))
        if target.dim()==2: target.unsqueeze_(0)
        #data_provider = self.df.iloc[idx].data_provider
        return im,target
 

    
def adjust_lr(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (1 -  (epoch // args.epochs))**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # dataset
    train_df = pd.read_csv(os.path.join(args.root, "train.csv"))
    is_radboud = (train_df['data_provider'] == 'radboud')
    by_radboud = train_df[is_radboud]
    #by_karolinska = train_df[np.logical_not(is_radboud)]
    df = {}
    
    df['train'], df['val'] = train_test_split(by_radboud, 
          stratify=by_radboud.isup_grade, test_size=0.02, random_state=42)
    dataset = {x: ProstateSeg(df[x], args.root, args.size, args.crop_size, x) 
                for x in ['train', 'val']}
    
    loader = {x: DataLoader(dataset[x],
                          batch_size=args.batch_size if x == 'train' else 1, 
                          shuffle = (x=='Train'),
                          num_workers = args.workers,
                          pin_memory=True)
                  for x in ['train', 'val']}
    
    
    model = models.segmentation.deeplabv3_resnet50(
    pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, nlabel)
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    for epoch in range(args.resume_epoch, args.epochs):
        adjust_lr(optimizer, epoch, args)
        for phase in ['train','val']:
            t0 = time.time()
            print("========={}:{}=========".format(phase,epoch))
            if phase == 'train':
                model.train()
            else:
                model.eval()
            num = 0
            correct = 0
            nums = np.zeros(6,dtype=int)
            corrects = np.zeros(6,dtype=int)
            running_loss=0
            for i, (inputs, masks) in enumerate(loader[phase]):
                inputs = inputs.to(device)     
                masks= masks.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs)
                    loss = criterion(output['out'], masks)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    num += inputs.size(0)
                    pred = output.argmax(dim=1)
                    #pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(masks).sum().item()
                    running_loss += loss.item() * inputs.size(0)
                    acc = 100.0 * correct / num
                    if (i+1) % args.log_step == 0:
                        s = "({},{:.1f}s) Loss:{:.3f} Acc:{:.3f}" 
                        print(s.format(num, (time.time()-t0)/(i+1), loss.item(), acc))
                    if phase == 'val':
                        for i in range(nlabel):
                            t = masks.eq(i)
                            nums[i] += t.sum().item()
                            corrects[i] += (pred.eq(i)&t).sum().item()
                            
            
            if epoch % 1 == 0 and phase=="train":
                torch.save(model.state_dict(), 
                           os.path.join(args.output_folder,"checkpoint-{}.pth".format(epoch)))
            if phase == 'val':
                print("|".join(["{}/{}".format(c,n) for c,n in zip(nums,corrects)]))

if __name__ == '__main__':
    main()


    