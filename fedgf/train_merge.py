from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, dirichlet_split_noniid, mnist_noniid_one,emnist_noniid
from models.gan import Generator_mnist, Discriminator_mnist, Generator_cifar, Discriminator_cifar
from models.Fed import FedAvg
from models.client import Client
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from utils.finetune_dataset import MyDataset
import numpy as np
import copy
import matplotlib.pyplot as plt
from math import fabs
import random
import matplotlib
from torch.autograd import Variable
from utils.sampling import mnist_noniid_one
from models.Nets import MLP, CNNMnist, CNNCifar, LeNet5
from models.test import test_img
from torch.utils.data import DataLoader, Dataset
#from utils.write import write_to_excel
import wandb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out

def seed_torch(seed=2):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
# 测试精度
def get_acc(net_glob, dataset_test, args):
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    return acc_test, loss_test
      
# 微调全局模型
def fine_tune(net_glob, g_list, args):
    # 根据输入，随机生成数据
    noise = torch.randn(64, 100, 1, 1).cuda()
    for i in range(len(g_list)):
        # 生成器生成伪数据
        fake = g_list[i](noise).detach().cpu().numpy()
        if i == 0:
            aug_data = fake
        else:
            aug_data=np.concatenate((aug_data,fake),axis=0)
    print(aug_data.shape)

    d = MyDataset(dataset=aug_data, num_per_generator=args.num_per_generator)
    dataloader = DataLoader(dataset=d, batch_size=args.bs, shuffle=True)
    net_glob.train()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr1, momentum=args.momentum)
    loss_func = nn.CrossEntropyLoss()
    # 微调
    for epoch in range(10):
        for idx, (images,labels) in enumerate(dataloader):
            images, labels = images.cuda(), labels.cuda()
            if args.dataset == 'cifar' or args.dataset == 'cinic':
                resize = transforms.Resize([32,32])
                images = resize(images)
            net_glob.zero_grad()
            log_probs = net_glob(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()  
    print("="*10,"fine-tune finished",'='*10)
    return net_glob
# 数据集处理函数
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
            '../../data/emnist/', train=True, download=True, transform=trains_femnist, split="byclass")
        dataset_test = datasets.EMNIST(
            '../../data/emnist/', train=False, download=True, transform=trains_femnist, split="byclass")
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Resize(64), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../../data/CIFAR', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../../data/CIFAR', train=False, download=True, transform=trans_cifar)
    elif args.dataset == 'cinic':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(),transforms.Resize(64), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.ImageFolder(root='../data/cinic/train', transform=trans_cifar)
        dataset_test = datasets.ImageFolder(root='../data/cinic/valid', transform=trans_cifar)
    print(len(dataset_train))
    print(len(dataset_test))
    return dataset_train, dataset_test

def get_model(args):
    if args.dataset == 'mnist' or args.dataset == 'fashionmnist' or args.dataset == 'emnist':
        net_glob = CNNMnist(args=args)
    elif args.dataset == 'cifar':
        net_glob = CNNCifar(args=args)
    return net_glob


if __name__ == '__main__':
    args = args_parser()
    print(args)
    seed_torch(args.seed)
    if args.use_wandb:
        config = {}
        for arg in vars(args):
            config.update({arg : getattr(args, arg)})
        wandb.init(project = args.project, config = config)
    if not os.path.exists('./testoutput/'):
        os.makedirs('./testoutput')
    # 获取训练，测试集
    dataset_train, dataset_test = get_dataset(args)
    if args.iid:
        if args.dataset == 'mnist' or args.dataset == 'fashionmnist':
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        # 狄利克雷划分
        if args.noniid_type == 'dirichlet':
            dict_users = dirichlet_split_noniid([dataset_train], args.num_users, args.alpha)
        else:
        # 病态noniid划分
            dict_users = mnist_noniid(args,dataset_train)
    
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
    
    class_num = len(dataset_train.classes)

    # 全局模型
    # net_glob = get_model(args).cuda()
    net_glob = LeNet5(args.dataset).cuda()
    net_glob.train()

    w_glob = net_glob.state_dict()
    loss_train = []
    cv_loss, cv_acc = [], []
    
    loss_test = []
    acc_test =[]

    # global 模型
    g_list = []
    d_list = []
    g_loss = [[] for _ in range(class_num)]
    d_loss = [[] for _ in range(class_num)]
    for i in range(class_num):
        if args.dataset == 'cifar' or args.dataset == 'cinic':
            netG = Generator_cifar(args).cuda()
            netD = Discriminator_cifar(args).cuda()
        else:
            netG = Generator_mnist(args).cuda()
            netD = Discriminator_mnist(args).cuda()
        netG.apply(weights_init)
        netD.apply(weights_init)
        g_list.append(netG)
        d_list.append(netD)

    label_set = set()

    for epoch in range(args.epochs):
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False).tolist()
        # ----- coder 记录列表 -----
        # ----- gan 记录列表 -----
        g_local_list = [[] for _ in range(class_num)]
        d_local_list = [[] for _ in range(class_num)]
        g_temp_loss = [[] for _ in range(class_num)]
        d_temp_loss = [[] for _ in range(class_num)]

        # ----- global 记录列表 -----
        loss_locals = []
        w_locals = []

        for client_id in idxs_users:
            # ----- gan 训练 ------
            # label = dataset_train[dict_users[client_id][0]][1]
            local = Client(args=args, dataset=dataset_train, idxs=dict_users[client_id], client_id=client_id, global_round= epoch)
            label =local.label
            label_set.add(label)
            # ----- 算法1：本地模型训练 开始------
            D, G, D_losses, G_losses = local.trainGan(D=copy.deepcopy(d_list[label]), G=copy.deepcopy(g_list[label]))
            g_local_list[label].append(G.state_dict())
            d_local_list[label].append(D.state_dict())

            g_temp_loss[label].append(G_losses)
            d_temp_loss[label].append(D_losses)

            w, loss = local.train(net=copy.deepcopy(net_glob).cuda())
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            # ----- 算法1：本地模型训练 结束------
            
        # ----- 算法2： 全局模型聚合 开始-----
        for i in range(class_num):
            if len(g_local_list[i]) != 0:
                newG = FedAvg(g_local_list[i])
                newD = FedAvg(d_local_list[i])
                g_list[i].load_state_dict(newG)
                d_list[i].load_state_dict(newD)

        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)
        # ----- 算法2： 全局模型聚合 结束----

        # ----- 算法3： 服务端更新 开始-----
        net_glob = fine_tune(copy.deepcopy(net_glob), g_list, args)
        # ----- 算法3： 服务端更新 结束-----
        
        # 测试
        net_glob.eval()
        res = get_acc(net_glob, dataset_test, args)
        net_glob.train()
        acc_test.append(res[0].item())
        loss_test.append(res[1])
        if args.use_wandb:
            wandb.log({'Average loss' : loss_avg,  'loss_test' : res[1], 'acc_test' : res[0].item()})
        
        # 保存模型
        if not os.path.exists("./testoutput/models/{}".format(epoch)):
            os.makedirs("./testoutput/models/{}".format(epoch))
        torch.save(net_glob.state_dict(), "./testoutput/models/{}/global_epoch{}.pth".format(epoch, epoch+1))       
        

    # gan 绘图
    for id in range(class_num):
        plt.figure()
        plt.plot(d_loss[id], label="D")
        plt.plot(g_loss[id], label="G")
        plt.legend()
        plt.savefig("./testoutput/loss"+str(id)+".png")
        torch.save(g_list[id].state_dict(), "./testoutput/models/generator"+str(id)+'.pth')        
        torch.save(d_list[id].state_dict(), "./testoutput/models/desciriminator"+str(id)+'.pth')
    
    # global绘图
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./testoutput/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))


    # 保存数据
    #write_to_excel(loss_test, acc_test, loss_train, args)

    print('='*20)
    print(label_set)

    # testing
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))
    # print("="*10,"training finished",'='*10)
    # get_acc(net_glob, dataset_train, args)
    

    # 开始全局微调
    # net_global = fine_tune(net_glob, g_list, args)
    # noise = torch.randn(64, 100, 1, 1).cuda()
    # for i in range(10):
    #     fake = g_list[i](noise).detach().cpu().numpy()
    #     if i == 0:
    #         aug_data = fake
    #     else:
    #         aug_data=np.concatenate((aug_data,fake),axis=0)
    # print(aug_data.shape)

    # d = MyDataset(dataset=aug_data, num_per_generator=args.num_per_generator)
    # dataloader = DataLoader(dataset=d, batch_size=args.bs, shuffle=True)
    # net_glob.train()
    # optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr1, momentum=args.momentum)
    # loss_func = nn.CrossEntropyLoss()
    # for epoch in range(10):
    #     for idx, (images,labels) in enumerate(dataloader):
    #         images, labels = images.cuda(), labels.cuda()
    #         net_glob.zero_grad()
    #         log_probs = net_glob(images)
    #         loss = loss_func(log_probs, labels)
    #         loss.backward()
    #         optimizer.step()  
    # print("="*10,"fine-tune finished",'='*10)
    # 结束全局微调

    # net_glob.eval()
    # get_acc(net_glob, dataset_train, args)


