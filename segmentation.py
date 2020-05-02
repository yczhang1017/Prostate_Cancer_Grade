import os
import time
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import random 
from random import uniform, randint
import openslide
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
 

parser = argparse.ArgumentParser(
    description='Prostate Cancer Grader')
parser.add_argument('--root', default='..',
                    type=str, help='directory of the data')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--epochs', default=25, type=int,
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
parser.add_argument('--log', default=1, type=int, help='steps to print log')


args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
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
nlabel = 6

def get_image_mask(
        img_id = "1d2f6f79ee18460e6b18e3710c87e8a2", data_dir = "..",
        size = 2048, crop_size = (600,600), mode='train'):
    image_path = os.path.join(data_dir, "train_images", img_id + '.tiff')
    mask_path = os.path.join(data_dir, "train_label_masks", img_id + '_mask.tiff')
    image = openslide.OpenSlide(image_path)
    mask = openslide.OpenSlide(mask_path)
    w0, h0 = image.dimensions
    r = np.sqrt(h0/w0)
    if mode == 'train':
        size = int(uniform(0.8,1.25)*size)
        ws, hs = int(size/r), int(size*r) 
        wc, hc = crop_size
        if wc > ws:
            size = size*wc//ws + 1
        if hc > hs:
            size = size*hc//hs + 1
            
        w1, h1 = int(size/r), int(size*r)
        x1, y1 = randint(0,w1-wc), randint(0,h1-hc)
        x0, y0 = w0*x1//w1, h0*y1//h1 
        k = image.get_best_level_for_downsample(w0//w1)
        wk, hk = image.level_dimensions[k]
        wck, hck = wc*wk//w1, hc*hk//h1
        im = image.read_region((x0,y0), k, (wck,hck)).convert('RGB')
        im = im.resize((wc,hc))      
        mm = mask.read_region((x0,y0), k, (wck,hck)).getchannel(0)
        mm = mm.resize((wc,hc))   
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
    def __init__(self, df, data_dir, size, crop, mode):
        self.df = df
        self.data_dir = data_dir
        self.size = size
        self.crop = crop
        self.mode = mode
       
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx].image_id
        im,mm = get_image_mask(img_id, self.data_dir, self.size, self.crop, self.mode)
        if transform is not None:
            im = transform(im)
        
        target = torch.from_numpy(np.array(mm).astype('int64'))
        return im,target
 
class FocalLoss(nn.Module):
    def __init__(self, alpha= None, gamma=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, x, y):
        x1 = F.log_softmax(x,1)
        nl = F.nll_loss(x1,y,reduction='none')
        pt = torch.exp(-nl)
        if self.alpha is None:
            return  ((1-pt)**self.gamma * nl).mean()
        nll = F.nll_loss(x1,y,weight=self.alpha,reduction='none')
        return ((1-pt)**self.gamma * nll).mean()

        
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
    num=0
    for idx, row in by_radboud.iterrows():
        img_id = row['image_id']
        mask_path = os.path.join(args.root,"train_label_masks",img_id+"_mask.tiff")
        if not os.path.isfile(mask_path):
            num += 1
            #print("{}:{} mask not work!".format(num,img_id))
            by_radboud = by_radboud.drop(idx)

    df = {} 
    df['train'], df['val'] = train_test_split(by_radboud, 
          stratify=by_radboud.isup_grade, test_size=20, random_state=42)
    
    dataset = {'val': ProstateSeg(df['val'], args.root, args.size, (args.crop_size, args.crop_size), 'val')}
    loader = {'val': DataLoader(dataset['val'],num_workers = args.workers,pin_memory=True)}
    model = models.segmentation.deeplabv3_resnet101(
            pretrained=(not args.checkpoint))
    
    model.classifier = DeepLabHead(2048, nlabel)
    model.to(device)
    if args.checkpoint:
        print('Resuming training from epoch {}, loading {}...'
              .format(args.resume_epoch,args.checkpoint))
        weight_file=os.path.join(args.output_folder, args.checkpoint)
        model.load_state_dict(torch.load(weight_file,
                                 map_location=lambda storage, loc: storage))
    
    
    criterion = FocalLoss(alpha = torch.tensor([1, 1.4, 8, 7, 6, 12],dtype=torch.float32,device=device))
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    for epoch in range(args.resume_epoch, args.epochs):
        s1, s2 = int(0.9*args.crop_size), int(1.1*args.crop_size)
        crop = (randint(s1,s2), randint(s1,s2))
        dataset['train'] = ProstateSeg(df['train'], args.root, args.size, crop, 'train')
        loader['train'] = DataLoader(dataset['train'],
            batch_size=args.batch_size, shuffle = True,
            num_workers = args.workers, pin_memory=True)
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
            pros = np.zeros(6,dtype=int)
            corrects = np.zeros(6,dtype=int)
            for i, (inputs, masks) in enumerate(loader[phase]):
                t1 = time.time()
                if i==0: print(inputs.shape) 
                inputs = inputs.to(device)   
                masks= masks.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs)
                    loss = criterion(output['out'], masks)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    num += masks.size(0)
                    npixel  = np.prod(masks.shape)
                    pred = output['out'].argmax(dim=1)
                    correct = pred.eq(masks).sum().item() 
                    acc = correct*100 / npixel
                    if (i+1) % args.log == 0:
                        t2 = time.time()
                        s = "({},{:.1f}s,{:.1f}s) Loss:{:.3f} Acc:{:.3f}" 
                        print(s.format(num, t2-t1, (t2-t0)/(i+1), loss.item(), acc))
                    if phase == 'val':
                        for i in range(nlabel):
                            t = masks.eq(i)
                            p = pred.eq(i)
                            nums[i] += t.sum().item()
                            pros[i] += p.sum().item()
                            corrects[i] += (p&t).sum().item()
                            
            
            if epoch % 1 == 0 and phase=="train":
                torch.save(model.state_dict(), 
                           os.path.join(args.output_folder,"deeplab-{}.pth".format(epoch)))
            if phase == 'val':
                print("recall:"+"|".join(["{:.5f}".format(c/n) for c,n in zip(corrects,nums)]))
                print("precision:"+"|".join(["{:.5f}".format(c/p) for c,p in zip(corrects,pros)]))

if __name__ == '__main__':
    main()


    
