import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

nlabel = 6


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    
class ResGrader(nn.Module):
    def __init__(self, arch, n =512, o=nlabel):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features 
        self.head = nn.Sequential(AdaptiveConcatPool2d(),Flatten())
        self.head2 = nn.Sequential(nn.Linear(2*nc,n),
                            Mish(),nn.BatchNorm1d(n), nn.Dropout(0.5),nn.Linear(512,o))
    def forward(self, x):
        b, n, c, w, h = x.shape
        x = x.view(b*n,c,w,h)
        #x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        bn, c, w, h = x.shape
        #x: bs*N x nc x 4 x 4
        x = x.view(b,n,c,w,h).permute(0,2,1,3,4).contiguous().view(b,c,w*n,h)
        x = self.head(x)
        #x: bs x 2nc 
        x = self.head2(x)
        #x: bs x o
        return x
        
        

class Grader(nn.Module):
    def __init__(self, arch, n=1000, o=nlabel):
        super(Grader, self).__init__()
        self.n = n
        self.model = EfficientNet.from_pretrained(arch)
        #self.model._fc = nn.Linear(self.model._fc.in_features, n-1)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm([25,n])
        #encoder_layer  = nn.TransformerEncoderLayer(n, 8)
        #self.attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc1 = nn.Linear(n,o)
        #self.norm2 = nn.LayerNorm([25,6])
        #self.fc2 = nn.Linear(25,1)
    def forward(self,x): # batch x 17 x size x size x 3
        b, n, c, w, h = x.shape
        x = self.model(x.view(b*n, c, w, h))
        x = x.view(b,n,-1)
        x = self.norm1(self.act(x)).view(b,n,-1)
        #x = self.attention(x)
        x = self.fc1(x) # b x 25 x o 
        #x = self.norm2(self.act(x))
        #x = self.fc2(x.permute(0,2,1)).squeeze()
        return x.mean(1)
"""
class Grader(nn.Module):
    def __init__(self, arch, n=256, o=nlabel):
        super(Grader, self).__init__()
        self.n = n
        self.model = EfficientNet.from_pretrained(arch)
        self.model._avg_pooling = AdaptiveConcatPool2d()
        nc = self.model._fc.in_features
        #self.norm1 = nn.LayerNorm([25,2*nc])
        self.act = Mish()
        self.model._fc = nn.Linear(nc*2+1, o) 
        #encoder_layer  = nn.TransformerEncoderLayer(n, 8)
        #self.attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm2 = nn.LayerNorm([25,o])
        self.fc2 = nn.Linear(25,1)
    def forward(self,x,p): # batch x 17 x size x size x 3
        b, n, c, w, h = x.shape
        x = self.model.extract_features(x.view(b*n, c, w, h))
        x = self.model._avg_pooling(x)
        x = x.view(b,n,-1)
        #x = self.norm1(self.act(x))
        p = p.expand((b,n)).unsqueeze(-1)
        x = torch.cat((x,p),dim=-1)
        x = self.model._dropout(x)
        x = self.model._fc(x) # b x 25 x o 
        #x = self.attention(x)
        x = self.norm2(self.act(x))
        x = self.fc2(x.permute(0,2,1)).squeeze()
        return x
"""    


    