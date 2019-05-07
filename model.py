import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_,constant_



def weight_inits(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_normal_(m.weight.data,mode='fan_out',nonlinearity='relu')
        # m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data=torch.eye(10).cuda()
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        constant_(m.weight.data, 1)
        constant_(m.bias.data, 0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)

class mnistNetwork(nn.Module):
    def __init__(self):
        super(mnistNetwork,self).__init__()

        self.backbone=nn.Sequential(
            #b*1*28*28
            nn.Conv2d(1,64,3,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #b*64*26*26
            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #b*64*24*24
            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            # b*64*11*11
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #b*128*9*9
            nn.Conv2d(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # b*128*7*7
            nn.Conv2d(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            # b*128*2*2
            nn.Conv2d(128,10,1,1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(10),
            #b*10*1*1
            Flatten(),
            nn.Linear(10,10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

        self.backbone.apply(weight_inits)

    def forward(self, x):
        return self.backbone(x)
