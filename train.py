import os
import argparse
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import time
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import random 
import openslide
import skimage.measure
import PIL
from PIL.ImageOps import invert

from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from efficientnet_pytorch import EfficientNet
from torch_multi_head_attention import MultiHeadAttention

    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

parser = argparse.ArgumentParser(
    description='Prostate Cancer Grader')
parser.add_argument('--root', default='..',
                    type=str, help='directory of the data')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('-w','--workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('-e','--epochs', default=15, type=int,
                    help='number of epochs to train')
parser.add_argument('-o','--output_folder', default='save/', type=str,
                    help='Dir to save results')
parser.add_argument('-wd','--weight_decay', default=2e-4, type=float,
                    help='Weight decay')
parser.add_argument('-c','--checkpoint', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('-r','--resume_epoch', default=0, type=int,
                    help='epoch number to be resumed at')
parser.add_argument('-s','--size', default=192, type=int,
                    help='image size for training, divisible by 64')
parser.add_argument('-ls','--log_step', default=10, type=int,
                    help='number of steps to print log')
parser.add_argument('--step', default=5, type=int,
                    help='step to reduce lr')
parser.add_argument('--fp16', action='store_false',
                    help='Run model fp16 mode.')

args = parser.parse_args()


if args.fp16 or args.distributed:
    try:
        from apex.fp16_utils import network_to_half,FP16_Optimizer
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")



if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cudnn.benchmark = True
else:
    device = torch.device("cpu")

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)
    

nlabel = 6
transform = {}
transform['train'] = transforms.Compose([
     transforms.RandomVerticalFlip(),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
     ])
transform['val'] = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
     ])
transform['test'] = transform['val'] 

def topk(X, n):
    x = np.zeros(n, dtype=int)
    y = np.zeros(n, dtype=int)
    x_ = np.ravel(X)
    for i in range(n):
        ij = x_.argmax()
        x_[ij] = 0
        x[i], y[i] = np.unravel_index(ij, X.shape)
    return x, y
'''
extract_images("001c62abd11fa4b57bf7a6c603a11bb9",
    "/kaggle/input/prostate-cancer-grade-assessment/train_images",
    256,
    False)
'''            
def extract_images(img_id, img_dir, size, debug):
    image_path = os.path.join(img_dir, img_id + '.tiff')
    image = openslide.OpenSlide(image_path)
    w0,h0 = image.level_dimensions[0]
    thumbnail = invert(image.get_thumbnail((size,size)))  
    img = np.array(thumbnail).mean(2)
    w1,h1 = thumbnail.size
    im = PIL.Image.new('RGB',(size,size))
    im.paste(thumbnail, (random.randrange(size+1-w1), random.randrange(size+1-h1)))
    num =  {16:8, 64:8}
    images = [im]
    for level, n in num.items():
        r = size // level
        label = skimage.measure.block_reduce(img, (r,r), np.mean)
        if debug:
            plt.figure()
            plt.imshow(label*255)

        xs,ys = topk(label,12)
        ll=list(range(12))
        random.shuffle(ll)
        ll = ll[:n]
        pts = [(x,y) for x,y in zip(xs[ll],ys[ll])]
        for x,y in pts:
            s0 = max(w0,h0)
            ix,iy = x*s0//level , y*s0//level
            im = image.read_region((iy,ix), 0, (s0//level,s0//level))        
            im = invert(im.resize((size,size)).convert('RGB'))
            images += [im]

    if debug:
        for im in images:
            plt.figure()
            plt.imshow(np.array(im))
    
    return images

class ProstateData(Dataset):
    def __init__(self, df, data_dir, mode, size, transform):
        self.img_dir = os.path.join(data_dir,"train_images")
        self.size = size
        self.mode = mode
        self.transform = transform
        self.df = df
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx].image_id
        label = self.df.iloc[idx].isup_grade
        images = extract_images(img_id, self.img_dir, self.size, False)
        image_tensor = None
        for im in images:
            tensor = self.transform(im).unsqueeze(0)
            if image_tensor is None:
                image_tensor = tensor
            else:
                image_tensor = torch.cat((image_tensor, tensor), dim=0)
            
        if self.mode == 'train' or self.mode == 'val':
            return image_tensor, torch.tensor(label, dtype=torch.long, device="cpu")
        else:
            return image_tensor


class Grader(nn.Module):
    def __init__(self, n = 64, o=nlabel):
        super(Grader, self).__init__()
        self.n = n
        self.models = [EfficientNet.from_pretrained('efficientnet-b0'),
                  EfficientNet.from_pretrained('efficientnet-b1'),
                  EfficientNet.from_pretrained('efficientnet-b1'),]
        self.fcq = [nn.Linear(1000,n),
                    nn.Linear(1000,n),
                    nn.Linear(1000,n)]
        self.fck = [nn.Linear(1000,n),
                    nn.Linear(1000,n),
                    nn.Linear(1000,n)]
        self.fcv = [nn.Linear(1000,n),
                    nn.Linear(1000,n),
                    nn.Linear(1000,n)]
        
        self.attention = MultiHeadAttention(in_features=n, head_num=8)
        self.fc1 = nn.Linear(n,o)
    def forward(self,x,size=args.size): # batch x 17 x size x size x 3
        b, n, c, w, h = x.shape
        xs = [x[:,0,:,:,:],
              x[:,1:9,:,:,:].reshape(b*8, c, w, h),
              x[:,9:,:,:,:].reshape(b*8, c, w, h)]
        q = None
        for x,m,fq,fk,fv in zip(xs, self.models, self.fcq, self.fck, self.fcv):
            y0 = m(x).reshape(b, -1, 1000)
            if q is None:
                q = fq(y0)
                k = fk(y0)
                v = fv(y0)
            else:
                q = torch.cat((q, fq(y0)), dim = 1)
                k = torch.cat((k, fk(y0)), dim = 1)
                v = torch.cat((v, fv(y0)), dim = 1)
        
        y = self.attention(q,k,v)   
        # y (b,17,n)
        y = self.fc1(y).mean(dim=1)
        return y
    
def main():
    train_csv = pd.read_csv(os.path.join(args.root, "train.csv"))
    df = {}
    df['train'], df['val'] = train_test_split(train_csv, test_size=0.05, random_state=42)
    dataset = {x: ProstateData(df[x], args.root, x, args.size, transform=transform[x]) 
                for x in ['train', 'val']}
    loader={x: DataLoader(dataset[x],
                          batch_size=args.batch_size, 
                          shuffle= (x=='Train'),
                          num_workers=args.workers,
                          pin_memory=True)
                  for x in ['train', 'val']}
    
    model = Grader()
    if torch.cuda.is_available():
        model=nn.DataParallel(model)
        
        
    if args.checkpoint:
        print('Resuming training from epoch {}, loading {}...'
              .format(args.resume_epoch,args.checkpoint))
        weight_file=os.path.join(args.root,args.checkpoint)
        model.load_state_dict(torch.load(weight_file,
                                 map_location=lambda storage, loc: storage))
    
    model.to(device)
    
    
    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
        model = network_to_half(model)
        
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
    
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer)
    for i in range(args.resume_epoch):
        scheduler.step()
    
    for epoch in range(args.resume_epoch, args.epochs):
        for phase in ['train','val']:
            t0 = time.time()
            print("=========",phase,"=========")
            if phase == 'train':
                model.train()
            else:
                model.eval()
            num = 0
            correct = 0
            nums = np.zeros(6,dtype=int)
            corrects = np.zeros(6,dtype=int)
            running_loss=0
            for i, (inputs, targets) in enumerate(loader[phase]):
                inputs = inputs.to(device).half() if args.fp16 else inputs.to(device)                 
                targets= targets.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs)
                    loss = criterion(output, targets)
                    if phase == 'train':
                        if args.fp16:
                            optimizer.backward(loss)
                        else:
                            loss.backward()
                        optimizer.step()
                    
                    num += inputs.size(0)
                    pred = output.argmax(dim=1)
                    #pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(targets).sum().item()
                    running_loss += loss.item() * inputs.size(0)
                    acc = 100.0 * correct / num
                    if (i+1) % args.log_step == 0:
                        s = "({},{:.1f}s) Loss:{:.3f} Acc:{:.3f}" 
                        print(s.format(num, (time.time()-t0)/(i+1), running_loss/num, acc))
                    
                    if phase == 'val':
                        s=""
                        for i in range(nlabel):
                            t = targets.eq(i)
                            nums[i] += t.sum().item()
                            corrects[i] = (pred.eq(i)&t).sum().item()
                            s+= "{}/{} |".format(nums[i],corrects[i])
                        print(s)
                        
            if phase=="train":scheduler.step()
            if epoch % 1 == 0 and phase=="train":
                torch.save(model.state_dict(), 
                           os.path.join(args.output_folder,"checkpoint-{}.pth".format(epoch)))

if __name__ == '__main__':
    main()
