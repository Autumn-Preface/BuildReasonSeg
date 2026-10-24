"""Plain same-padded U-Net, 64/128/256/512/1024, no pretrained components."""
import torch
from torch import nn


def double_conv(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,3,padding=1),nn.ReLU(inplace=True),
                         nn.Conv2d(out_channels,out_channels,3,padding=1),nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths=[64,128,256,512,1024]
        self.down=nn.ModuleList([double_conv(3,64)]+[double_conv(a,b) for a,b in zip(widths,widths[1:])])
        self.pool=nn.MaxPool2d(2)
        self.up=nn.ModuleList([nn.ConvTranspose2d(a,b,2,stride=2) for a,b in zip(widths[:0:-1],widths[-2::-1])])
        self.fuse=nn.ModuleList([double_conv(2*b,b) for b in widths[-2::-1]])
        self.out=nn.Conv2d(64,2,1)
    def forward(self,x):
        if x.shape[-1]%16 or x.shape[-2]%16:raise ValueError('U-Net requires multiples of 16')
        skips=[]
        for i,block in enumerate(self.down):
            x=block(x if i==0 else self.pool(x));skips.append(x)
        for up,fuse,skip in zip(self.up,self.fuse,skips[-2::-1]):x=fuse(torch.cat((up(x),skip),dim=1))
        return self.out(x)


def loss_function(logits,target,valid):
    """0.5 masked BCE + 0.5 per-known-channel Soft Dice; epsilon 1e-6."""
    import torch.nn.functional as F
    if not valid.any():raise ValueError('batch has no supervised pixels')
    mask=valid.to(logits.dtype);target=target.to(logits.dtype)
    bce=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*mask).sum()/mask.sum()
    prob=logits.sigmoid();axes=(0,2,3)
    numerator=2*(prob*target*mask).sum(axes)+1e-6
    denominator=((prob+target)*mask).sum(axes)+1e-6
    active=mask.sum(axes)>0
    dice=(1-numerator[active]/denominator[active]).mean()
    return 0.5*bce+0.5*dice
