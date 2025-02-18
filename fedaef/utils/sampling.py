#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import torch
from torchvision import datasets, transforms
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from collections import Counter


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(args, dataset):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    print(args.dataset)
    if args.dataset == 'cifar' or args.dataset == 'cifar10':
        num_shards, num_imgs = 100, 500
    elif args.dataset == 'cinic' :
        num_shards, num_imgs = 100, 900
    else:
        if args.num_users == 50:
            num_shards, num_imgs = 100, 600
        else:
            num_shards, num_imgs = 200, 300
    # if num_users == 100: # mnist,fsmnist, emnist
    #     num_shards, num_imgs = 200, 300
    # else: # cifar
    #     num_shards, num_imgs = 100, 500
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
    idxs = np.arange(num_shards * num_imgs)
    if args.num_users == 100:
        labels = dataset.train_labels.numpy()
    else:
        labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 按垂直方向（行顺序）堆叠数组构成一个新的数组
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 将idxs_labels按照标签的大小进行排序
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(args.num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))   # 随机选取两个切片
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # 将两个切片包含的数据集加入数组
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# def  dirichlet_split_noniid(dataset, alpha, num_users):
#     '''
#     参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
#     '''
#     train_labels = dataset.targets
#     if not torch.is_tensor(train_labels):
#         train_labels = torch.tensor(train_labels)
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     # n_classes = train_labels.max()+1
#     n_classes = len(dataset.classes)
#     label_distribution = np.random.dirichlet([alpha]*num_users, n_classes)
#     # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

#     class_idcs = [np.argwhere(train_labels==y).flatten() 
#            for y in range(n_classes)]
#     # 记录每个K个类别对应的样本下标
 
#     client_idcs = [[] for _ in range(num_users)]
#     # 记录N个client分别对应样本集合的索引
#     for c, fracs in zip(class_idcs, label_distribution):
#         # np.split按照比例将类别为k的样本划分为了N个子集
#         # for i, idcs 为遍历第i个client对应样本集合的索引
#         for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
#             client_idcs[i] += [idcs]

#     client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
#     for i in range(num_users):
#         dict_users[i] = client_idcs[i]
  
#     return dict_users

def dirichlet_split_noniid(
    ori_dataset: List[Dataset],
    num_clients: int,
    alpha: float,
    transform=None,
    target_transform=None,
):
    print(type(alpha))
    NUM_CLASS = len(ori_dataset[0].classes)
    MIN_SIZE = 0
    X = [[] for _ in range(num_clients)]
    Y = [[] for _ in range(num_clients)]
    stats = {}
    targets_numpy = np.concatenate(
        [ds.targets for ds in ori_dataset], axis=0, dtype=np.int64
    )
    data_numpy = np.concatenate(
        [ds.data for ds in ori_dataset], axis=0, dtype=np.float32
    )
    print(data_numpy.shape)
    idx = [np.where(targets_numpy == i)[0] for i in range(NUM_CLASS)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    while MIN_SIZE < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(NUM_CLASS):
            np.random.shuffle(idx[k])
            distributions = np.random.dirichlet(np.repeat(alpha, num_clients))
            distributions = np.array(
                [
                    p * (len(idx_j) < len(targets_numpy) / num_clients)
                    for p, idx_j in zip(distributions, idx_batch)
                ]
            )
            distributions = distributions / distributions.sum()
            distributions = (np.cumsum(distributions) * len(idx[k])).astype(int)[:-1]
            idx_batch = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(idx_batch, np.split(idx[k], distributions))
            ]

            MIN_SIZE = min([len(idx_j) for idx_j in idx_batch])

        
        for i in range(num_clients):
            stats[i] = {"x": None, "y": None}
            np.random.shuffle(idx_batch[i])
            X[i] = data_numpy[idx_batch[i]]
            Y[i] = targets_numpy[idx_batch[i]]
            stats[i]["x"] = len(X[i])
            stats[i]["y"] = Counter(Y[i].tolist())
            dict_users[i] = idx_batch[i]

    return dict_users

def mnist_noniid_one(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 100, 600
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 按垂直方向（行顺序）堆叠数组构成一个新的数组
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 将idxs_labels按照标签的大小进行排序
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))   # 随机选取1个切片
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # 将1个切片包含的数据集加入数组
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def emnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 3450
        
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    if num_users == 100:
        labels = dataset.train_labels.numpy()[0:len(idxs)]
    else:
        labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 按垂直方向（行顺序）堆叠数组构成一个新的数组
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 将idxs_labels按照标签的大小进行排序
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))   # 随机选取两个切片
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # 将两个切片包含的数据集加入数组
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
