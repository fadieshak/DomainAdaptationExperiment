"""
Data handling script.
This is different from 'data_handler.py' as this one loads the
pre-computed clustering pseudo-labeles for the target dataset.

Modified from https://github.com/ayushtues/ADDA_pytorch
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MNISTWithSoftClusters(torchvision.datasets.MNIST):
    """
    Definition of MNIST dataset where we add the soft pseudo labels and return them in the __getitem__() function.
    """
    def __init__(self, *args, soft_labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_labels = soft_labels

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        soft_label = self.soft_labels[index]

        return img, target, soft_label

# Inititializing the used transforms for the different datasets.
transform_mnist = transforms.Compose(
    [
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.Resize([28,28]),
    transforms.ToTensor()
    ])
transform_mnist_test = transforms.Compose(
    [transforms.Resize([28,28]),
    transforms.ToTensor()
    ])

transform_svhn_train = transforms.Compose([
    transforms.Resize([28,28]),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.Grayscale(), 
    transforms.ToTensor()
    ]) 
transform_svhn_test = transforms.Compose([
    transforms.Resize([28,28]),
    transforms.Grayscale(), 
    transforms.ToTensor()
    ]) 

transform_usps_test = transforms.Compose(
    [transforms.Resize([28,28]),
    transforms.ToTensor()
    ])        
transform_usps = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.Resize([28,28]),
    transforms.ToTensor()
    ])

# Loading soft pseudo-labels.
soft_clusters = torch.load('./datasets/soft_clusters.pt') 
# Initializing datasets.
mnist_data_train = MNISTWithSoftClusters(root='./datasets', train=True, transform=transform_mnist, download=True, soft_labels=soft_clusters)

svhn_data_train = torchvision.datasets.SVHN('./datasets', split='train', transform=transform_svhn_train, target_transform=None, download=True)
usps_data_train = torchvision.datasets.USPS('./datasets', train=True, transform=transform_usps, target_transform=None, download=True)

mnist_data_test = torchvision.datasets.MNIST('./datasets', train=False, transform=transform_mnist_test, target_transform=None, download=True)
svhn_data_test = torchvision.datasets.SVHN('./datasets', split='test', transform=transform_svhn_test, target_transform=None, download=True)
usps_data_test = torchvision.datasets.USPS('./datasets', train=False, transform=transform_usps_test, target_transform=None, download=True)

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

# Initializing different benchmark pairings tested in the experiments.
svhn_mnist_train = ConcatDataset(svhn_data_train, mnist_data_train)
mnist_usps_train = ConcatDataset(mnist_data_train, usps_data_train)
usps_mnist_train = ConcatDataset(usps_data_train, mnist_data_train)


def get_dataloader_svhn_mnist_train(batch_size):
    print("LEN OF MNIST  :",len(mnist_data_train))
    print("LEN OF SVHN  :",len(svhn_data_test))
    print("LEN OF SVHN + MNIST dataset :",len(svhn_mnist_train))
    dataloader_svhn_mnist_train = torch.utils.data.DataLoader(svhn_mnist_train,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_svhn_mnist_train

def get_dataloader_mnist_usps_train(batch_size):
    print("LEN OF MNIST  :",len(mnist_data_train))
    print("LEN OF USPS  :",len(usps_data_train))
    print("LEN OF MNIST + USPS dataset :",len(mnist_usps_train))
    dataloader_mnist_usps_train = torch.utils.data.DataLoader(mnist_usps_train,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_mnist_usps_train

def get_dataloader_usps_mnist_train(batch_size):
    print("LEN OF MNIST  :",len(mnist_data_train))
    print("LEN OF USPS  :",len(usps_data_train))
    print("LEN OF MNIST + USPS dataset :",len(usps_mnist_train))
    dataloader_usps_mnist_train = torch.utils.data.DataLoader(usps_mnist_train,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_usps_mnist_train

def get_dataloader_mnist_train(batch_size):
    dataloader_mnist_train = torch.utils.data.DataLoader(mnist_data_train,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_mnist_train


def get_dataloader_mnist_test(batch_size):
    dataloader_mnist_test = torch.utils.data.DataLoader(mnist_data_test,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_mnist_test


def get_dataloader_svhn_train(batch_size):
    dataloader_svhn_train = torch.utils.data.DataLoader(svhn_data_train,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_svhn_train


def get_dataloader_svhn_test(batch_size):
    dataloader_svhn_test = torch.utils.data.DataLoader(svhn_data_test,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_svhn_test



def get_dataloader_usps_train(batch_size):
    dataloader_usps_train = torch.utils.data.DataLoader(usps_data_train,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_usps_train



def get_dataloader_usps_test(batch_size):
    dataloader_usps_test = torch.utils.data.DataLoader(usps_data_test,batch_size=batch_size,drop_last=True,shuffle=True)
    return dataloader_usps_test
