from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,TensorDataset
import torch

def get_mnist(args):
    train_data=MNIST(root='./data',train=True,download=True)
    test_data=MNIST(root='./data',train=False,download=True)


    data=torch.cat([train_data.data.float()/255,test_data.data.float()/255],0)
    labels=torch.cat([train_data.targets,test_data.targets],0)

    ds=TensorDataset(data,labels)

    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True)

    return dl



