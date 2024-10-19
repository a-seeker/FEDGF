import random
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import OrderedDict, Union

import numpy as np
import torch
from torchvision import datasets, transforms
import os
import copy
from utils.sampling import mnist_noniid,dirichlet_split_noniid, mnist_iid, cifar_iid
from torch.utils.data import DataLoader, Dataset

def fix_random_seed(seed: int) -> None:
    # torch.cuda.empty_cache()
    # torch.random.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
# 数据集处理
def get_dataset(args):
    print("="*20,args.dataset)
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_train = datasets.MNIST(
            '../../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../../data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'fashionmnist':
        trains_femnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_train = datasets.FashionMNIST(
            '../../data/FMNIST/', train=True, download=True, transform=trains_femnist)
        dataset_test = datasets.FashionMNIST(
            '../../data/FMNIST/', train=False, download=True, transform=trains_femnist)
    elif args.dataset == 'emnist':
        trains_femnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_train = datasets.EMNIST(
            '../data/emnist/', train=True, download=True, transform=trains_femnist, split="byclass")
        dataset_test = datasets.EMNIST(
            '../data/emnist/', train=False, download=True, transform=trains_femnist, split="byclass")
    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../../data/cifar/', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../../data/cifar/', train=False, download=True, transform=trans_cifar)

    print(args.noniid_type)
    if args.iid:
        if args.dataset == 'mnist' or args.dataset == 'fashionmnist':
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        if args.noniid_type == 'dirichlet':
            dict_users = dirichlet_split_noniid([dataset_train], args.num_users, args.alpha)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    # 22222222222
    l_sum = [0 for _ in range(10)]
    for k,v in dict_users.items():
        l = [0]*10
        m = 0
        m_id = 0
        for j in v:
            idx = int(dataset_train[j][1])
            l[idx] += 1
            if l[idx] > m:
                m = l[idx]
                m_id = idx
        l_sum[m_id] += 1
        print(m,''*10, l)
    print(l_sum)
    # 22222222222
    
    print(len(dataset_train))
    print(len(dataset_test))
    return dataset_train, dataset_test, dict_users

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def clone_parameters(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad) for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )
