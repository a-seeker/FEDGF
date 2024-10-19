import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
from utils.util import DatasetSplit

class FedProxLocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.device = torch.device("cuda" if self.args.gpu > -1 and torch.cuda.is_available() else "cpu")

    def train(self, net):
        global_model = copy.deepcopy(net)
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                
                # -------mode 1---------
                # proximal_term = 0.0
                # for w, w_t in zip(net.parameters(), global_model.parameters()):
                #     proximal_term += (w - w_t).norm(2)
                # loss = self.loss_func(log_probs, labels) + (0.1 / 2) * proximal_term

                # --------mode 2----------
                proxy = 0
                loss = self.loss_func(log_probs, labels)
                for p_g, p_l in zip(global_model.parameters(), net.parameters()):
                    proxy = proxy + torch.sum((p_g - p_l) * (p_g - p_l))
                proxy = (0.01 / 2) * proxy
                loss = loss + proxy


                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
