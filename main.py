from model import mnistNetwork
from dataloader import get_mnist
import argparse
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch.optim import Adam,SGD,RMSprop
import os
import numpy as np
from torch.utils.data import DataLoader,TensorDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


parse=argparse.ArgumentParser('DAC')
parse.add_argument('--batch_size',type=int,default=128)
args=parse.parse_args()

dl=get_mnist(args)

model=mnistNetwork()
model=model.cuda()

Lambda=0

opti_model=RMSprop(model.parameters(),lr=0.001)

epoch=1
u=0.95
l=0.455
while u>l:

    u=0.95-Lambda
    l=0.455+0.1*Lambda

    model.train()
    i=1
    while i <1001:
        for x,_ in dl:
            if x.size(0)<args.batch_size:
                break
            x=x.view(-1,1,28,28).cuda()
            f=model(x)
            f_norm=F.normalize(f,p=2,dim=1)
            I=f_norm.mm(f_norm.t())

            loss=-torch.mean((I.detach()>u).float()*torch.log(torch.clamp(I,1e-10,1))+(I.detach()<l).float()*torch.log(torch.clamp(1-I,1e-10,1)))

            opti_model.zero_grad()
            loss.backward()
            opti_model.step()



            if i%20==0:
                print('[Epoch {}]\t[Iteration {}]\t[Loss={:.4f}]'.format(epoch,i,loss.detach().cpu().numpy()))

            i+=1
            if i==1001:
                break




    model.eval()
    pre_y=[]
    tru_y=[]
    i=0
    for x,y in dl:
        x=x.view(-1,1,28,28).cuda()
        f=model(x)
        pre_y.append(torch.argmax(f,1).detach().cpu().numpy())
        tru_y.append(y.numpy())
        i+=1
        if i==10:
            break

    pre_y=np.concatenate(pre_y,0)
    tru_y=np.concatenate(tru_y,0)

    print('[ACC={:.4f}]\t[NMI={:.4f}]\t[ARI={:.4f}]'.format(ACC(tru_y,pre_y),NMI(tru_y,pre_y),ARI(tru_y,pre_y)))




    Lambda+=1.1*0.009
    epoch+=1

