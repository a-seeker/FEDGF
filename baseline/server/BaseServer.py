import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import random
from torchvision import datasets, transforms
import torch
import wandb

import sys
sys.path.append(os.path.split(sys.path[0])[0])
from utils.util import fix_random_seed
from models.Nets import LeNet5
# from models.Update import FedAVGLocalUpdate, FedProxLocalUpdate
from client.FedAvgClient import FedAVGLocalUpdate
from models.test import test_img
from utils.write import write_to_excel

class BaseServer:
    def __init__(self, args, dataset_train, dataset_test, dict_users) -> None:
        self.args = args
        self.device = torch.device("cuda" if self.args.gpu > -1 and torch.cuda.is_available() else "cpu")
        #fix_random_seed(args.seed)
        self.backbone = LeNet5
        self.global_epochs = args.epochs
        self.args = args
        self.net_glob = LeNet5(args.dataset).cuda()
        self.w_glob = self.net_glob.state_dict()

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users

        # 训练损失、测试损失、测试精度
        self.loss_train, self.loss_list, self.acc_list= [], [], [] 

    def train(self):
        self.net_glob.train()
        if self.args.use_wandb:
            config = {}
            for arg in vars(self.args):
                config.update({arg : getattr(self.args, arg)})
            wandb.init(project = self.args.project, config = config)
        # cv_loss, cv_acc = [], []
        # val_loss_pre, counter = 0, 0
        # net_best = None
        # best_loss = None
        # val_acc_list, net_list = [], []
        if self.args.all_clients: 
            print("Aggregation over all clients")
            w_locals = [self.w_glob for i in range(self.args.num_users)]
        for iter in range(self.args.epochs):
            loss_locals = []
            if not self.args.all_clients:
                w_locals = []
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
            for idx in idxs_users:
                local = FedAVGLocalUpdate(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(self.net_glob).to(self.device))
                if self.args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                
                loss_locals.append(copy.deepcopy(loss))

            # update global weights
            self.aggregate(w_locals)

            self.test()
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            if self.args.use_wandb:
                wandb.log({'Average loss' : loss_avg,  'loss_test' : self.loss_list[-1], 'acc_test' : self.acc_list[-1]})
            self.loss_train.append(loss_avg)

        self.drawPic()


    def aggregate(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))

        self.w_glob = w_avg
        # copy weight to net_glob
        self.net_glob.load_state_dict(self.w_glob)

    def test(self):
        self.net_glob.eval()
        acc_test, loss_test = test_img(self.net_glob, self.dataset_test, self.args)
        self.acc_list.append(acc_test.item())
        self.loss_list.append(loss_test)
        self.net_glob.train()

    def drawPic(self):
        # record in excel    
        write_to_excel(self.args,self.loss_list, self.acc_list, self.loss_train)

        # # plot loss curve
        # plt.figure()
        # plt.plot(range(len(self.loss_train)), self.loss_train)
        # plt.ylabel('train_loss')
        # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(self.args.dataset, self.args.model, self.args.epochs, self.args.frac, self.args.iid))

        # # testing
        # self.net_glob.eval()
        # acc_train, loss_train = test_img(self.net_glob, self.dataset_train, self.args)
        # acc_test, loss_test = test_img(self.net_glob, self.dataset_test, self.args)
        # print("Training accuracy: {:.2f}".format(acc_train))
        # print("Testing accuracy: {:.2f}".format(acc_test))

