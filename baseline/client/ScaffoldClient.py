from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict

import torch
from torch import nn, autograd
from utils.util import DatasetSplit
from torch.utils.data import DataLoader, Dataset
from models.Nets import LeNet5

class SCAFFOLDClient:
    def __init__(self, args, dataset=None):
        self.args = args
        self.device = torch.device("cuda" if self.args.gpu > -1 and torch.cuda.is_available() else "cpu")
        print('-'*10,self.device)
        self.c_local: Dict[List[torch.Tensor]] = {}
        self.c_diff = []
        self.untrainable_params: Dict[str, Dict[str, torch.Tensor]] = {}
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.model = LeNet5(args.dataset).cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)

        

    def train(self, client_id: int, model_params: OrderedDict[str, torch.Tensor], c_global, idxs):
        self.ldr_train = DataLoader(DatasetSplit(self.dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.client_id = client_id
        self.set_parameters(model_params)

        if self.client_id not in self.c_local.keys():
            self.c_diff = c_global
        else:
            self.c_diff = []
            for c_l, c_g in zip(self.c_local[self.client_id], c_global):
                self.c_diff.append(-c_l + c_g)
        
        epoch_loss = self._train()
        # update local control variate
        with torch.no_grad():
            trainable_parameters = filter(
                lambda p: p.requires_grad, model_params.values()
            )

            if self.client_id not in self.c_local.keys():
                self.c_local[self.client_id] = [
                    torch.zeros_like(param, device=self.device)
                    for param in self.model.parameters()
                ]

            y_delta = []
            c_plus = []
            c_delta = []

            # compute y_delta (difference of model before and after training)
            for param_l, param_g in zip(self.model.parameters(), trainable_parameters):
                y_delta.append(param_l - param_g)

            # compute c_plus
            # coef = 1 / (self.args.local_ep * self.args.lr)
            coef = 1 / (len(self.ldr_train)*self.args.local_bs * self.args.lr)
            for c_l, c_g, diff in zip(self.c_local[self.client_id], c_global, y_delta):
                c_plus.append(c_l - c_g - coef * diff)

            # compute c_delta
            for c_p, c_l in zip(c_plus, self.c_local[self.client_id]):
                c_delta.append(c_p - c_l)

            self.c_local[self.client_id] = c_plus

        if self.client_id not in self.untrainable_params.keys():
            self.untrainable_params[self.client_id] = {}
        for name, param in self.model.state_dict(keep_vars=True).items():
            if not param.requires_grad:
                self.untrainable_params[self.client_id][name] = param.clone()

        return (y_delta, c_delta), sum(epoch_loss) / len(epoch_loss)

    def set_parameters(self, model_params: OrderedDict):
        self.model.load_state_dict(model_params, strict=False)
        if self.client_id in self.untrainable_params.keys():
            self.model.load_state_dict(
                self.untrainable_params[self.client_id], strict=False
            )

    def _train(self):
        epoch_loss = []
        self.model.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                # self.model.zero_grad()
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                for param, c_d in zip(self.model.parameters(), self.c_diff):
                    param.grad += c_d.data
                self.optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return epoch_loss


