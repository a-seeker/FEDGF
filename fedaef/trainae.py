from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, dirichlet_split_noniid, mnist_noniid_one,emnist_noniid
from models.Fed import FedAvg, FedAvgByWeight
from models.coder import AutoEncoder, AutoEncoderCifar
from models.client import Client
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from utils.finetune_dataset import MyDataset, LoadData
import numpy as np
import copy
import matplotlib.pyplot as plt
from math import fabs
import random
import matplotlib
from utils.sampling import mnist_noniid_one
from models.Nets import MLP, CNNMnist, CNNCifar, LeNet5, resnet18
from models.test import test_img, test_img_tc
from torch.utils.data import DataLoader, Dataset
from utils.write import write_to_excel
from utils.generate import generatePic, loadFakeData
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

def get_acc(net_glob, dataset_test, args):
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    return acc_test, loss_test
      

def fine_tune(net_glob, coder_list, enc_list, args):
    noise_factor = 0.5
    for i in range(10):
        if len(enc_list[i]) == 0:
            continue
        aug_data = None
        c = coder_list[i]
        encode = enc_list[i]
        #print(encode)
        for j in range(4):
            encode = encode + noise_factor * np.random.normal(loc=0, scale=1)
            #print(encode.shape) // 16,3
            #print(c.decoder(encode).shape) 16,784
            if args.dataset == 'mnist' or args.dataset == 'fashionmnist':
                out = c.decoder(encode).view(16, 1,28,28).detach().cpu()
            else:
                out = c.decoder(encode).detach().cpu()
                
            if aug_data is None:
                aug_data = out
            else:
                aug_data=np.concatenate((aug_data, out),axis=0)
        #print(aug_data.shape)
        #torch.save(aug_data, './save/finetune_'+str(i)+'.pt')

    d = MyDataset(dataset=aug_data, num_per_generator=args.num_per_generator)
    dataloader = DataLoader(dataset=d, batch_size=args.bs, shuffle=True)
    
    
    net_glob.train()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=0.001, momentum=args.momentum)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(args.finetune_ep):
        for idx, (images,labels) in enumerate(dataloader):
            images, labels = images.cuda(), labels.cuda()
            if args.dataset == 'cifar' or args.dataset == 'cinic':
                resize = transforms.Resize([32,32])
                images = resize(images)
            logits = net_glob(images)
            loss = loss_func(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
    print("="*10,"fine-tune finished",'='*10)
    
    return net_glob

    
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
            '../../data/FMNIST', train=True, download=True, transform=trains_femnist)
        dataset_test = datasets.FashionMNIST(
            '../../data/FMNIST', train=False, download=True, transform=trains_femnist)
    elif args.dataset == 'emnist':
        trains_femnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_train = datasets.EMNIST(
            '../data/emnist/', train=True, download=True, transform=trains_femnist, split="byclass")
        dataset_test = datasets.EMNIST(
            '../data/emnist/', train=False, download=True, transform=trains_femnist, split="byclass")
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Resize(32), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../../data/cifar', train=False, download=True, transform=trans_cifar)
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

def print_split_data(dict_users):
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

    dataset_train, dataset_test = get_dataset(args)
    if args.iid:
        if args.dataset == 'mnist' or args.dataset == 'fashionmnist':
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        if args.noniid_type == 'dirichlet':
            dict_users = dirichlet_split_noniid([dataset_train], args.num_users, args.alpha)
        else:
            dict_users = mnist_noniid(args, dataset_train)
        
    print_split_data(dict_users)
    
    class_num = len(dataset_train.classes)

    # 全局模型
    # net_glob = get_model(args).cuda()
    net_glob = LeNet5(args.dataset).cuda()
    net_glob.train()
    
    # 教师模型

    w_glob = net_glob.state_dict()
    loss_train = []
    cv_loss, cv_acc = [], []
    
    loss_test = []
    acc_test =[]
    
    loss_test_tc, acc_test_tc = [], []

    # global 模型
    coder_list = []
    coder_loss = [[] for _ in range(class_num)]
    for i in range(class_num):
        if args.dataset == 'cifar' or args.dataset == 'cinic':
            netEncoder = AutoEncoderCifar().cuda()
        else:
            netEncoder = AutoEncoder().cuda()
        netEncoder.apply(weights_init)
        coder_list.append(netEncoder)
    label_set = set()
    enc_list = [[] for _ in range(class_num)]
    for epoch in range(args.epochs):
        idxs_users = np.random.choice(range(args.num_users), 10, replace=False).tolist()
        # ----- gan 记录列表 -----
        coder_local_list = [[] for _ in range(class_num)]
        coder_temp_loss = [[] for _ in range(class_num)]
        weights1 = []
        weights2 = [[] for _ in range(class_num)]
        # ----- global 记录列表 -----
        loss_locals = []
        w_locals = []
        new_enc_list = [[] for _ in range(class_num)]
        for client_id in idxs_users:
            # ----- coder 训练 ------
            # label = dataset_train[dict_users[client_id][0]][1]
            local = Client(args=args, dataset=dataset_train, idxs=dict_users[client_id], client_id=client_id, global_round= epoch)
            label =local.label
            label_set.add(label)
            coder, coder_losses, enc = local.trainCoder(copy.deepcopy(coder_list[label]))
            if coder != None:
                coder_local_list[label].append(coder.state_dict())
                coder_temp_loss[label].append(coder_losses)
                new_enc_list[label].append(copy.deepcopy(enc))
                weights2[label].append(len(local.dataloader2))
                
            if not os.path.exists('./save/'+str(epoch)):
                os.makedirs('./save/'+str(epoch))
                
            # ----- 全局模型训练 ------
            w, loss = local.train(net=copy.deepcopy(net_glob).cuda())
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weights1.append(len(local.dataloader1))
        
        # ----- coder 聚合 -----
        for i in range(class_num):
            if len(coder_local_list[i]) != 0:
                newCoder = FedAvgByWeight(coder_local_list[i], weights2[i])
                coder_list[i].load_state_dict(newCoder)
                newEnc = sum(new_enc_list[i]) / len(new_enc_list[i])
                enc_list[i] = newEnc
        # if epoch <= 100:
        #     for i in range(class_num):
        #         torch.save(coder_list[i].state_dict(), './save/{}/coder{}.pt'.format(epoch, i))
        #     torch.save(enc_list, './save/{}/enc_list.pt'.format(epoch))
        # ----- fedavg聚合 -----
        w_glob = FedAvgByWeight(w_locals, weights1)
        net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)

        # 微调
        #if epoch <=10 or epoch % 10 == 0:
        net_glob = fine_tune(copy.deepcopy(net_glob), coder_list, enc_list, args)
        net_glob.eval()
        print("fine-tune acc")
        res = get_acc(net_glob, dataset_test, args)
        net_glob.train()
        acc_test.append(res[0].item())
        loss_test.append(res[1])
        
        if args.use_wandb:
            wandb.log({'Average loss' : loss_avg,  'loss_test' : res[1], 'acc_test' : res[0].item()})
        
        # 保存模型
        if epoch <= 100:
            if not os.path.exists("./testoutput/models/{}".format(epoch)):
                os.makedirs("./testoutput/models/{}".format(epoch))
            torch.save(net_glob.state_dict(), "./testoutput/models/{}/global_epoch{}.pth".format(epoch, epoch+1))
            for z in range(class_num):
                torch.save(coder_list[z].state_dict(), "./testoutput/models/{}/generator_{}_epoch{}.pth".format(epoch, z, epoch+1))        
        

    # 保存数据
    write_to_excel(loss_test, acc_test, loss_train, args)

    print('='*20)
    print(label_set)
