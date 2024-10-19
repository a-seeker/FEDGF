import os
import numpy as np
import torch
import copy
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
ARGS = {
    "mnist": 28,
    "cifar": 64,
    "fashionmnist": 28
}

class MnistDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.idx = idxs
        self.dataset = dataset

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        return self.dataset[self.idx[index]][0], self.dataset[self.idx[index]][1]


def get_client_label(dataset, idxs):
    class_num = len(dataset.classes)
    label_dict = {i:[] for i in range(class_num)}
    for index in idxs:
        label = dataset[index][1]
        label_dict[label].append(index)
    
    mx_len = 0
    mx_label = 0
    for k,v in label_dict.items():
        if(len(v) > mx_len):
            mx_len = len(v)
            mx_label = k

    d = MnistDataset(dataset=dataset, idxs=np.array(label_dict[mx_label]))
    return mx_label, d



class Client(object):
    def __init__(self, args, dataset, idxs, client_id, global_round):
        self.args = args
        self.client_id = client_id  # 客户端id
        self.global_round = global_round  # 当前是第几轮
        self.dataset = dataset
        d = MnistDataset(dataset=dataset, idxs=idxs)
        self.dataloader1 = DataLoader(dataset=d, batch_size=args.local_bs1, shuffle=True) # local train数据集

        self.label, d = get_client_label(dataset, idxs)
        self.dataloader2 = DataLoader(dataset=d, batch_size=args.local_bs2, shuffle=True, drop_last = True)   # gan train数据集

    # 使用fmnist数据集的旧模型使用该训练
    def trainCoder(self, coder):
        print(len(self.dataloader2) * self.args.local_bs2)
        if len(self.dataloader2) < 1:
            return None, None, None
        print(str(self.client_id)+ " Starting Training local coder...")
        coder.train()
        optimizer = torch.optim.Adam(coder.parameters(),lr=self.args.lr2)
        if self.args.dataset == 'mnist' or self.args.dataset == 'fashionmnist':
            loss_func = nn.MSELoss()
        else:
            loss_func = nn.L1Loss()
        epoch_loss = []
        img_sz = ARGS[self.args.dataset]
        encoder_list = []
        for epoch in range(self.args.local_ep2):
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.dataloader2):
                x, y = x.cuda(), y.cuda()
                if self.args.dataset == 'mnist' or self.args.dataset == 'fashionmnist':
                    b_x = x.view(-1, img_sz*img_sz)
                    b_y = x.view(-1, img_sz*img_sz)
                    b_label = y
                else:
                    b_x = x
                    b_y = x
                encode, decode = coder(b_x)
                if epoch == self.args.local_ep2 - 1:
                    encoder_list.append(encode.clone().detach())
                loss = loss_func(decode, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return coder.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(encoder_list) / len(encoder_list)
    
    # l2正则化
    def l2_regularization(model, l2_alpha=0.001):
        l2_loss = []
        for module in model.modules():
            if type(module) is nn.Conv2d:
                l2_loss.append((module.weight ** 2).sum() / 2.0)
        return l2_alpha * sum(l2_loss)
    
    # 稀疏正则化
    def KL_divergence(self, p, q):
        """
        Calculate the KL-divergence of (p,q)
        :param p:
        :param q:
        :return:
        """
        q = torch.nn.functional.softmax(q, dim=0)  # 首先要用softmax对隐藏层结点输出进行归一化处理
        q = torch.sum(q, dim=0)/16  # dim:缩减的维度,q的第一维是batch维,即大小为batch_size大小,此处是将第j个神经元在batch_size个输入下所有的输出取平均
        s1 = torch.sum(p*torch.log(p/q))
        s2 = torch.sum((1-p)*torch.log((1-p)/(1-q)))
        return s1+s2

    # coder模型为卷积模型
    def trainCoder2(self, coder, preTrain = False):
        expect_tho = 0.05
        tho_tensor = torch.FloatTensor([expect_tho for _ in range(128)]).cuda()
        print(len(self.dataloader2) * self.args.local_bs2)
        if len(self.dataloader2) < 1:
            return None, None, None
        print(str(self.client_id)+ " Starting Training local coder...")
        coder.train()
        optimizer = torch.optim.Adam(coder.parameters(),lr=self.args.lr2)
        loss_func = nn.MSELoss()
        epoch_loss = []
        img_sz = ARGS[self.args.dataset]
        encoder_list = []
        if preTrain:
            print("preTrain start epoch = ", self.args.local_ep2)
            local_ep2 = self.args.local_ep2
        else:
            local_ep2 = self.args.local_ep1
        for epoch in range(local_ep2):
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.dataloader2):
                x, y = x.cuda(), y.cuda()
                encode, decode = coder(x)
                # 提取最后一个batch的特征
                if epoch == local_ep2 - 1:
                    encoder_list.append(encode.clone().detach())
                # 稀疏正则
                _kl = self.KL_divergence(tho_tensor, encode)
                loss = loss_func(decode, x)+ _kl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        #path = './save/'+str(self.client_id)
        #if not os.path.exists(path):
        #    os.mkdir(path)
        #torch.save(encoder_list,path+'/encode_list.pt')
        return coder.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(encoder_list) / len(encoder_list)

    def train(self, net):
        print(str(self.client_id)+ " Starting Training local model...")
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr1, momentum=self.args.momentum)
        loss_func = nn.CrossEntropyLoss()
        epoch_loss = []
        for iter in range(self.args.local_ep1):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.dataloader1):
                images, labels = images.cuda(), labels.cuda()
                if self.args.dataset == 'cifar' or self.args.dataset == 'cinic':
                    resize = transforms.Resize([32,32])
                    images = resize(images)

                net.zero_grad()
                log_probs = net(images)
                # if self.client_id == 22:
                #     print(labels)
                #     print(log_probs)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.dataloader1.dataset),
                               100. * batch_idx / len(self.dataloader1), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    


    
    def loss_function(self, x_hat, x, mu, log_var):
        """
        Calculate the loss. Note that the loss includes two parts.
        :param x_hat:
        :param x:
        :param mu:
        :param log_var:
        :return: total loss, BCE and KLD of our model
        """
        # 1. the reconstruction loss.
        # We regard the MNIST as binary classification
        BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')

        # 2. KL-divergence
        # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
        # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
        KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)

        # 3. total loss
        loss = BCE + KLD
        return loss, BCE, KLD
    
    def kl_divergence(self, mu, logvar):
        kld = -0.5*torch.sum(1+logvar-mu**2-torch.exp(logvar), dim=-1)
        return kld.mean()