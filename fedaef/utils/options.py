#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    # 客户端选择概率
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    # 本地模型epoch
    parser.add_argument('--local_ep1', type=int, default=5, help="the number of local epochs: E")
    # 本地AE epoch
    parser.add_argument('--local_ep2', type=int, default=20, help="the number of local gan epochs: E")
    parser.add_argument('--finetune_ep', type=int, default=5, help="the number of finetune epochs: E")
     # 本地模型batchsize
    parser.add_argument('--local_bs1', type=int, default=10, help="local batch size: B")
    # 本地AE batchsize
    parser.add_argument('--local_bs2', type=int, default=16, help="local gan batch size: B")
    parser.add_argument('--bs', type=int, default=16, help="test batch size")
    # 本地模型lr
    parser.add_argument('--lr1', type=float, default=0.01, help="learning rate")
    # 本地AE lr
    parser.add_argument('--lr2', type=float, default=0.0002, help="gan learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--allocate', action='store_true', help='use allocate method')
    parser.add_argument('--image_size', type=int, default=28, help='Spatial size of training images')
    parser.add_argument('--nz', type=int, default=100, help='Size of z latent vector (i.e. size of generator input)')
    parser.add_argument('--ngf', type=int, default=28, help='Size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=28, help='Size of feature maps in discriminator')
    parser.add_argument('--beta1', type=int, default=0.5, help='Beta1 hyperparam for Adam optimizers')
    parser.add_argument('--num_per_generator', type=int, default=64, help='sample num of each generator')
    # 方法名
    parser.add_argument('--flmethod', type=str, default='fedavg', help='algorithm of fl')
    # noniid划分方式
    parser.add_argument('--noniid_type', type=str, default='fedavg', help='noniid type (fedavg or dirichlet)')
    # 狄利克雷系数
    parser.add_argument('--alpha', type=float, default=0.3, help='dirichlet ratio')
    # 是否使用wandb记录数据
    parser.add_argument('--use_wandb', action= 'store_true', default=False, help='use wandb store')
    parser.add_argument('--project', type=str, default='fedgf', help='wandb project')
    parser.add_argument('--save_gan_pic', action= 'store_true', default=False, help='save gan local train pic')
    parser.add_argument('--rep_dim', type=int, default=20, help='input z shape')
    args = parser.parse_args()
    return args
