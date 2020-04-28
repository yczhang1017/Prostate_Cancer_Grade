import os
import argparse
import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
from tqdm import tqdm

import PIL
os.environ['PATH'] = "C:/ProgramData/Anaconda3/lib/site-packages/openslide/bin" + ";" + os.environ['PATH']
print(os.environ['PATH'] )
import openslide
import skimage.measure
from PIL.ImageOps import invert

from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler

from efficientnet_pytorch import EfficientNet
from torch_multi_head_attention import MultiHeadAttention


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser(
    description='Prostate Cancer Grader')
parser.add_argument('--root', default='./',
                    type=str, help='directory of the data')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('-w','--workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('-lr', '--learning-rate', default=0.01, type=float,
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
parser.add_argument('-s','--size', default=256, type=int,
                    help='image size for training')

args = parser.parse_args()
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cpu")

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
    
def getp1(img):
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i,:].mean()>0 or img[:,j].mean()>0: 
                return j,i
def getp2(img):
    h,w = img.shape
    for i in range(h-1,-1,-1):
        for j in range(w-1,-1,-1):
            if img[i,:].mean()>0 or img[:,j].mean()>0: 
                return i,j
            
transform = transforms.Compose([
     transforms.RandomVerticalFlip(),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
     ])

'''
extract_images("001c62abd11fa4b57bf7a6c603a11bb9",
    "/kaggle/input/prostate-cancer-grade-assessment/train_images",
    256,
    False)
'''            
def extract_images(img_id,img_dir,size,debug):
    img_path = os.path.join(img_dir, img_id+'.tiff')
    image = openslide.OpenSlide(img_path)
    w0,h0 = image.level_dimensions[0]
    out = size
    s1 = size
    thumbnail = image.get_thumbnail((s1,s1))
    img = np.array(thumbnail)
    img = 255 - img.mean(axis=2)
    
    i1,j1=getp1(img)
    i2,j2=getp2(img)
    box1 = np.array([j1,i1,j2,i2]) 
    h1,w1=img.shape   #size before crop
    h2,w2=i2-i1,j2-j1 #size after crop

    s3 = s1*out//max(w2,h2)
    box3 = box1*out//max(w2,h2)
    im1 = image.get_thumbnail((s3,s3)).crop(box3)
    w3, h3 = im1.size
    x = random.randrange(out-w3) if w3<out else 0
    y = random.randrange(out-h3) if h3<out else 0
    im = PIL.Image.new("RGB", (out,out), (0,0,0))
    im.paste(invert(im1), box = (x,y))
    
    
    num =  {16:8, 64:8}
    images = [im]
    for level, n in num.items():
        r = out // level
        label = skimage.measure.block_reduce(img, (r,r), np.mean)
        label = label > 20
        if debug:
            plt.figure()
            plt.imshow(label*255)

        xs,ys = label.nonzero()
        pts = [(x,y) for x,y in zip(xs,ys)]
        random.shuffle(pts)
        for j in range(n):
            x,y = pts[j]
            s0 = max(w0,h0)
            ix,iy = x*s0//level , y*s0//level
            im = image.read_region((iy,ix), 0, (s0//level,s0//level))        
            im = invert(im.resize((out,out)).convert('RGB'))
            images += [im]

    if debug:
        for im in images:
            plt.figure()
            plt.imshow(np.array(im))
    
    return images

class ProstateData(Dataset):
    def __init__(self, df, image_dir, mode, size, transform):
        self.img_dir = image_dir
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
            
        if self.mode == 'train' or self.mode == 'valid':
            return image_tensor, torch.tensor(label, dtype=torch.long)
        else:
            return image_tensor


class Grader(nn.Module):
    def __init__(self, n = 64, o=6):
        super(Grader, self).__init__()
        self.n = n
        self.models = [EfficientNet.from_pretrained('efficientnet-b0'),
                  EfficientNet.from_pretrained('efficientnet-b0'),
                  EfficientNet.from_pretrained('efficientnet-b0'),]
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
    def forward(self,x,size=256): # batch x 17 x size x size x 3
        b, n, c, w, h = x.shape
        xs = [x[:,0,:,:,:],
              x[:,1:9,:,:,:].view(b*8, c, w, h),
              x[:,9:,:,:,:].view(b*8, c, w, h)]
        q = None
        for x,m,fq,fk,fv in zip(xs, self.models, self.fcq, self.fck, self.fcv):
            y0 = m(x).view(b, -1, 1000)
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
    


def train(model,
    lr = 0.1,
    momentum = 0.9,
    weight_decay = 1e-4,
    start = 0 ,
    end = 15,
    batch_size = 1,
    checkpoint = None,
    size =256,
    num_workers = 4,
    use_tpu = False,
    data_dir = "/kaggle/input/prostate-cancer-grade-assessment/",
    log_steps =1,
):
    
    data, pred, target = None, None, None
    for i in range(start): 
        scheduler.step()
    for epoch in tqdm(range(start,end), desc="Epoch", disable=disable_logging):
        for phase in ['train','val']:
            if phase == 'train':
                loader = pl.ParallelLoader(train_loader, [device]) if use_tpu else train_loader
                model.train()
                scheduler.step()
            else:
                loader = pl.ParallelLoader(test_loader, [device]) if use_tpu else test_loader
                model.eval()
            iterator = tqdm(loader, desc=phase, total=len(loader), disable=disable_logging)
            total_samples = 0
            correct = 0
            with torch.set_grad_enabled(phase == 'train'):
                for i, (inputs, target) in enumerate(iterator):
                    optimizer.zero_grad()
                    output = model(inputs)
                    loss = criterion(output, target)
                    loss.backward()
                    if phase == 'train':
                        if use_tpu:
                            xm.optimizer_step(optimizer)
                        else:
                            optimizer.step()
                        scheduler.step()
            
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total_samples += inputs.size()[0]
                    accuracy = 100.0 * correct / total_samples
                    if i % log_steps == 0:
                        print('({}) Loss:{:.3f} Acc:{:.3f}'.format(
                             total_samples, loss.item(), accuracy)) #xm.get_ordinal()
            if epoch % 5 == 0:
                torch.save(model.state_dict(), "checkpoint-{}.pth".format(epoch))
 
def main():
    #data_dir = "/kaggle/input/prostate-cancer-grade-assessment/"    
    data_dir = "data"
    train_csv = pd.read_csv(os.path.join(data_dir,"train.csv"))
    train_df, vali_df = train_test_split(train_csv, test_size=0.05, random_state=42)
    train_dir = os.path.join(data_dir,"train_images")
    dataset = {'train': ProstateData(train_df, train_dir, 'train', 256, transform),
               'vali': ProstateData(vali_df, train_dir, 'vali', 256, transform)}
    train_sampler = RandomSampler(dataset['train'])
    train_loader = DataLoader(
        dataset['train'],
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True)
    vali_loader = DataLoader(
        dataset['vali'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer)
    accuracy = 0.0
    model = Grader() 
    train(model)    


if __name__ == '__main__':
    main()