import copy
import numpy as np
import wandb
from BaseServer import BaseServer
from utils.util import get_dataset
from utils.options import args_parser
from client.FedProxClient import FedProxLocalUpdate
import numpy as np
import torch
import os
import random

class FedProxServer(BaseServer):
    def __init__(self, args, dataset_train, dataset_test, dict_users) -> None:
        super().__init__(args, dataset_train, dataset_test, dict_users)
    def train(self):
        if self.args.use_wandb:
            config = {}
            for arg in vars(self.args):
                config.update({arg : getattr(self.args, arg)})
            wandb.init(project = self.args.project, config = config)
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
                local = FedProxLocalUpdate(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[idx])
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
            self.loss_train.append(loss_avg)
            
            if self.args.use_wandb:
                wandb.log({'Average loss' : loss_avg,  'loss_test' : self.loss_list[-1], 'acc_test' : self.acc_list[-1]})
        self.drawPic()
        

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
    server = FedProxServer(args=args, dataset_train=dataset_train, dataset_test=dataset_test, dict_users= dict_users)
    server.train()

# python -u server/FedProxServer.py  --dataset mnist  --num_channels 1 --model cnn --epochs 200 --gpu 0 --num_users 100 --flmethod fedprox > log_mnist_fedprox.txt 2>&1 &
