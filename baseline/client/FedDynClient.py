import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
import copy
from utils.util import DatasetSplit

class FedDynLocalUpdate(object):
    def __init__(self, args, model, dataset=None, idxs=None):
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.device = torch.device("cuda" if self.args.gpu > -1 and torch.cuda.is_available() else "cpu")

        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.server_state_dict = None
        
        self.prev_grads = None
        for param in self.model.parameters():
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = torch.zeros_like(param.view(-1))
            else:
                self.prev_grads = torch.cat((self.prev_grads, torch.zeros_like(param.view(-1))), dim=0)

    def train(self, global_net):
        self.model = global_net
        self.server_state_dict = copy.deepcopy(self.model.state_dict())
        self.optim = torch.optim.SGD(self.model.parameters(),
                         lr=self.learning_rate,
                         weight_decay=1e-4)
        self.model.train()
        # train and update

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optim.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels.cuda())

                 #=== Dynamic regularization === #
                # Linear penalty
                lin_penalty = 0.0
                curr_params = None
                for name, param in self.model.named_parameters():
                    if not isinstance(curr_params, torch.Tensor):
                        curr_params = param.view(-1)
                    else:
                        curr_params = torch.cat((curr_params, param.view(-1)), dim=0)
                lin_penalty = torch.sum(curr_params * self.prev_grads)
                loss -= lin_penalty


                quad_penalty = 0.0
                for name, param in self.model.named_parameters():
                    quad_penalty += F.mse_loss(param, self.server_state_dict[name], reduction='sum')

                loss += self.alpha/2.0 * quad_penalty
                loss.backward()
                batch_loss.append(loss.item())

                # Update the previous gradients
                self.prev_grads = None
                for param in self.model.parameters():
                    if not isinstance(self.prev_grads, torch.Tensor):
                        self.prev_grads = param.grad.view(-1).clone()
                    else:
                        self.prev_grads = torch.cat((self.prev_grads, param.grad.view(-1).clone()), dim=0)

                self.optim.step()
            epoch_loss.append(sum(batch_loss)/len(batch_loss)) 
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)
