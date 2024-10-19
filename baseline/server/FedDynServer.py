import copy
import torch
import numpy as np
from BaseServer import BaseServer
from utils.util import get_dataset
from utils.options import args_parser
from client.FedDynClient import FedDynLocalUpdate
from collections import OrderedDict
from typing import OrderedDict, Union
from models.test import test_img
from utils.write import write_to_excel

class FedDynServer(BaseServer):
    def __init__(self, args, dataset_train, dataset_test, dict_users) -> None:
        super().__init__(args, dataset_train, dataset_test, dict_users)

        self.h = self.net_glob.state_dict().copy()
        self.clients = []
        for p in range(self.args.num_users):
            self.clients.append(FedDynLocalUpdate(args,copy.deepcopy(self.net_glob),self.dataset_train,self.dict_users[p]))

    def train(self):
        self.net_glob.train()
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
                w, loss = self.clients[idx].train(copy.deepcopy(self.net_glob))
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
            self.loss_train.append(loss_avg)
        self.drawPic()

    def aggregate(self, w_locals):
        print(f"Server: Updating model...")
        num_participants = len(w_locals)
        sum_theta = w_locals[0]
        for client_theta in w_locals[1:]:
            for key in client_theta.keys():
                sum_theta[key] += client_theta[key]

        delta_theta = {}
        for key in self.net_glob.state_dict().keys():
            delta_theta[key] = sum_theta[key] - self.net_glob.state_dict()[key]

        for key in self.h.keys():
            self.h[key] -= self.alpha * (1./self.num_users) * delta_theta[key]

        for key in self.net_glob.state_dict().keys():
            self.net_glob.state_dict()[key] = (1./num_participants) * sum_theta[key] - (1./self.alpha) *  self.h[key]
        print("Server: Updated model.")




    def clone_parameters(self, src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]) -> OrderedDict[str, torch.Tensor]:
        if isinstance(src, OrderedDict):
            return OrderedDict(
                {
                    name: param.clone().detach().requires_grad_(param.requires_grad)
                    for name, param in src.items()
                }
            )
        if isinstance(src, torch.nn.Module):
            return OrderedDict(
                {
                    name: param.clone().detach().requires_grad_(param.requires_grad)
                    for name, param in src.state_dict(keep_vars=True).items()
                }
            )


if __name__ == "__main__":
    args = args_parser()
    print(args)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    server = FedDynServer(args=args, dataset_train=dataset_train, dataset_test=dataset_test, dict_users= dict_users)
    server.train()
