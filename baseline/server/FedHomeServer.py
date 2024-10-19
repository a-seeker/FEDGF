import copy
import numpy as np
from BaseServer import BaseServer
from utils.util import get_dataset
from utils.options import args_parser
from client.FedHomeClient import FedHomeClient
from models.Nets import LeNet5_FedHome

class FedHomeServer(BaseServer):
    def __init__(self, args, dataset_train, dataset_test, dict_users) -> None:
        super().__init__(args, dataset_train, dataset_test, dict_users)
        self.clients = []
        self.net_glob = LeNet5_FedHome(args.dataset).cuda()
        self.w_glob = self.net_glob.state_dict()
        self.clients = []
        self.set_clients(args, FedHomeClient)

    def train(self):
        self.done = False
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
                # local = FedProxLocalUpdate(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[idx])
                local = self.clients[idx]
                w, loss = local.train(net=copy.deepcopy(self.net_glob).to(self.device), model_params=copy.deepcopy(self.w_glob),
                    dataset=self.dataset_train, idxs=self.dict_users[idx])
                if self.args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                
                loss_locals.append(copy.deepcopy(loss))
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            self.loss_train.append(loss_avg)

        # update global weights
        self.aggregate(w_locals)
        for idx in range(self.args.num_users):
            local = self.clients[idx]
            local.generate_data(dataset=self.dataset_train, idxs=self.dict_users[idx])

        for i in range(20):
            for client in self.clients:
                client.train_pred()
            self.test()
            


       
        self.drawPic()

    def set_clients(self, args, clientObj):
        for i in range(range(args.num_users)):
            client = clientObj(args, id=i)
            self.clients.append(client)   

    def test(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.set_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
        ids = [c.id for c in self.selected_clients]
        return ids, num_samples, tot_correct, tot_auc

if __name__ == "__main__":
    args = args_parser()
    print(args)
    dataset_train, dataset_test, dict_users = get_dataset(args)
    server = FedHomeServer(args=args, dataset_train=dataset_train, dataset_test=dataset_test, dict_users= dict_users)
    server.train()

# python -u server/FedProxServer.py  --dataset mnist  --num_channels 1 --model cnn --epochs 200 --gpu 0 --num_users 100 --flmethod fedprox > log_mnist_fedprox.txt 2>&1 &
