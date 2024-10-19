import copy
import torch
import numpy as np
from BaseServer import BaseServer
from utils.util import get_dataset
from utils.options import args_parser
from client.ScaffoldClient import SCAFFOLDClient
from collections import OrderedDict
from typing import OrderedDict, Union
from models.test import test_img
from utils.write import write_to_excel
import wandb
import os
import random
class ScaffoldServer(BaseServer):
    def __init__(self, args, dataset_train, dataset_test, dict_users) -> None:
        super().__init__(args, dataset_train, dataset_test, dict_users)
        self.global_lr = 1.0
        self.c_global = [
           torch.zeros_like(param).to(self.device)
           for param in self.net_glob.parameters()
        ]
        self.global_params_dict = OrderedDict(
                self.net_glob.state_dict(keep_vars=True)
            )
        self.trainer = SCAFFOLDClient(args=self.args, dataset=self.dataset_train)

    def train(self):
        if self.args.use_wandb:
            config = {}
            for arg in vars(self.args):
                config.update({arg : getattr(self.args, arg)})
            wandb.init(project = self.args.project, config = config)
        self.net_glob.train()
        for iter in range(self.args.epochs):
            loss_locals = []
            res_cache = []
            m = max(int(self.args.frac * self.args.num_users), 1)
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
            for idx in idxs_users:
                client_local_params = self.clone_parameters(self.global_params_dict)
                res, loss = self.trainer.train(client_id=idx, model_params=client_local_params, c_global=self.c_global, idxs=self.dict_users[idx])
                res_cache.append(res)
                loss_locals.append(copy.deepcopy(loss))

            # update global weights
            self.aggregate(res_cache)

            self.test()

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            self.loss_train.append(loss_avg)
            if self.args.use_wandb:
                wandb.log({'Average loss' : loss_avg,  'loss_test' : self.loss_list[-1], 'acc_test' : self.acc_list[-1]})
        self.drawPic()

    def aggregate(self, res_cache):
        y_delta_cache = list(zip(*res_cache))[0] 
        c_delta_cache = list(zip(*res_cache))[1]
        trainable_parameter = filter(
            lambda param: param.requires_grad, self.global_params_dict.values()
        )
        client_num_per_round = int(self.args.frac * self.args.num_users)
        avg_weight = torch.tensor(
            [
                1 / client_num_per_round
                for _ in range(client_num_per_round)
            ],
            device=self.device,
        )
        for param, y_del in zip(trainable_parameter, zip(*y_delta_cache)):
            x_del = torch.sum(avg_weight * torch.stack(y_del, dim=-1), dim=-1)
            param.data += self.global_lr * x_del
        # update global control
        for c_g, c_del in zip(self.c_global, zip(*c_delta_cache)):
            c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
            c_g.data += (
                client_num_per_round / self.args.num_users
            ) * c_del 




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

    def test(self):
        self.net_glob.load_state_dict(self.global_params_dict, strict=False)
        self.net_glob.eval()
        acc_test, loss_test = test_img(self.net_glob, self.dataset_test, self.args)
        self.acc_list.append(acc_test.item())
        self.loss_list.append(loss_test)
        self.net_glob.train()
            

if __name__ == "__main__":
    args = args_parser()
    print(args)
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    dataset_train, dataset_test, dict_users = get_dataset(args)
    server = ScaffoldServer(args=args, dataset_train=dataset_train, dataset_test=dataset_test, dict_users= dict_users)
    server.train()
